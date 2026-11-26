# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Semantics of the Hexagon DMA queue operators.

The operators are ``tirx.hexagon.dma_push``, ``dma_pop`` and ``dma_flush``.

They are registered in ``src/backend/hexagon/op/dma.cc`` and written in
TVMScript as ``T.hexagon.dma_push(...)``. On the hexagon target, LowerIntrin
turns them into calls to ``tirx_hexagon_dma_push/pop/flush`` in the C runtime
shim ``src/backend/hexagon/runtime/tirx_hexagon_dma.h``.
This module describes each call as an event of the model below and checks it
against the hardware limits, so that a race checker can consume the events.

Semantics follow the per-thread DMA queue that the reference Qualcomm
operators in llama.cpp use (``ggml/src/ggml-hexagon/htp/dma-queue.h``):

* Every hardware thread owns exactly one DMA queue.  The operators never name
  the queue; they always act on the queue of the thread that executes them.
* ``dma_push`` appends one asynchronous 1D/2D copy to the tail of the queue and
  returns immediately.
* ``dma_pop`` blocks until the *oldest* outstanding copy has completed and
  removes it from the queue (strict FIFO).
* ``dma_flush`` blocks until every outstanding copy has completed and empties
  the queue.  It is equivalent to calling ``dma_pop`` until the queue is empty.

Event model (for a race checker)
--------------------------------------------

Actors are hardware threads ``T`` and the DMA job ``J`` that each push creates.
Write ``T.k`` for the k-th event of thread ``T`` in program order.

``dma_push`` at ``T.k`` creates job ``J`` with sequence number ``n`` (the
number of earlier pushes by ``T``).
  * ``J`` reads ``nrows`` rows of ``row_size`` bytes from ``src``, row ``r``
    starting at ``src + r * src_stride``.
  * ``J`` writes the same shape to ``dst``, row ``r`` starting at
    ``dst + r * dst_stride``.
  * Edge ``T.k -> J.start``: memory written by ``T`` before the push is visible
    to the copy (llama.cpp issues ``release`` before ``dmlink``).
  * No edge from ``J`` to any later event of ``T``.  Until the matching pop,
    every access of ``T`` to the source or destination region is a race
    candidate.
  * No edge between two jobs of the same queue.  Completions are *observed* in
    FIFO order, but the model does not assume that the transfers themselves
    are serialized, so two in-flight jobs that write overlapping bytes, or where
    one writes what the other reads, race.
  * ``nrows == 0`` is a dummy push: it occupies a queue slot and must be popped,
    but touches no memory.  llama.cpp uses it to recycle buffer slots.

``dma_pop`` at ``T.k`` consumes the oldest outstanding job ``J``.
  * Edge ``J.end -> T.k``: everything ``J`` wrote is visible to ``T`` after the
    pop, and ``J`` no longer touches its regions.
  * Popping an empty queue is a protocol error.  (llama.cpp silently returns a
    zero ``dma_data``; agent-written code that does this is always a bug.)

``dma_flush`` at ``T.k`` consumes every outstanding job, oldest first, with an
edge ``J.end -> T.k`` for each.  Flushing an empty queue is allowed.

Protocol errors: popping an empty queue; more than ``capacity`` outstanding
jobs (llama.cpp's push fails when the ring is full); jobs still outstanding
when the thread finishes the kernel.

Unlike llama.cpp, ``dma_pop`` returns nothing.  llama.cpp returns the job's
``(dst, src)`` pair so that a pipeline can recycle buffer slots; in TIRx the
FIFO order makes the popped job known statically, so the program indexes its
slots itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import tvm_ffi

from tvm_tirx import ir

#: Hardware descriptor limits for v75 and newer (24-bit fields; nrows is 16-bit).
MAX_NROWS = 0xFFFF
MAX_ROW_SIZE = 0x00FFFFFF
MAX_STRIDE = 0x00FFFFFF

#: Queue depth used by llama.cpp (``dma_queue_init(..., 256, ...)``).  The
#: ring keeps one slot free, so at most ``capacity - 1`` jobs are outstanding.
DEFAULT_QUEUE_CAPACITY = 256

PUSH_ARGS = ("dst", "src", "dst_stride", "src_stride", "row_size", "nrows")

DMA_PUSH = ir.Op.get("tirx.hexagon.dma_push")
DMA_POP = ir.Op.get("tirx.hexagon.dma_pop")
DMA_FLUSH = ir.Op.get("tirx.hexagon.dma_flush")
DMA_OPS = (DMA_PUSH, DMA_POP, DMA_FLUSH)


def is_dma_call(expr) -> bool:
    """Whether ``expr`` is a call to one of the three DMA operators."""
    return isinstance(expr, ir.Call) and any(expr.op.same_as(op) for op in DMA_OPS)


# ---------------------------------------------------------------------------
# Event description, shared by validation, lowering and the race checker.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TensorRef:
    """A pointer argument resolved to the tensor element it addresses."""

    tensor: ir.Expr
    indices: tuple
    scope: str
    dtype: str


@dataclass(frozen=True)
class Access2D:
    """``nrows`` rows of ``row_size`` bytes, ``stride`` bytes apart, starting at ``base``."""

    base: TensorRef | None  # None when the pointer is not T.address_of(tensor[...])
    pointer: ir.Expr
    row_size: ir.Expr
    nrows: ir.Expr
    stride: ir.Expr


@dataclass(frozen=True)
class DmaEvent:
    """What one DMA call does, in the terms of the module docstring."""

    kind: str  # "push", "pop" or "flush"
    call: ir.Call
    reads: Access2D | None = None  # what the job reads (push only)
    writes: Access2D | None = None  # what the job writes (push only)

    @property
    def is_dummy(self) -> bool:
        """Whether this is a push with a constant ``nrows`` of 0."""
        return self.kind == "push" and _const_int(self.writes.nrows) == 0


def _const_int(expr):
    if isinstance(expr, int):
        return expr
    value = getattr(expr, "value", None)
    return value if isinstance(value, int) else None


def resolve_pointer(ptr) -> TensorRef | None:
    """Resolve ``T.address_of(tensor[indices])`` to the tensor element; otherwise None."""
    if not (isinstance(ptr, ir.Call) and getattr(ptr.op, "name", "") == "tirx.address_of"):
        return None
    if len(ptr.args) != 1 or not isinstance(ptr.args[0], ir.TensorLoad):
        return None
    load = ptr.args[0]
    ty = load.source.ty
    return TensorRef(
        tensor=load.source,
        indices=tuple(load.indices),
        scope=str(getattr(ty, "scope", "global")),
        dtype=str(getattr(ty, "dtype", load.ty.dtype)),
    )


def describe(call) -> DmaEvent:
    """Describe one DMA call as an event of the model in this module's docstring."""
    if call.op.same_as(DMA_POP):
        return DmaEvent("pop", call)
    if call.op.same_as(DMA_FLUSH):
        return DmaEvent("flush", call)
    if not call.op.same_as(DMA_PUSH):
        raise ValueError(f"not a DMA call: {call.op.name}")
    dst, src, dst_stride, src_stride, row_size, nrows = call.args
    return DmaEvent(
        "push",
        call,
        reads=Access2D(resolve_pointer(src), src, row_size, nrows, src_stride),
        writes=Access2D(resolve_pointer(dst), dst, row_size, nrows, dst_stride),
    )


def validate(call, *, allow_raw_pointers: bool = False) -> DmaEvent:
    """Check a DMA call against the hardware limits and the model; return its event.

    Checks that need values are applied only to arguments that are constants.

    Raises
    ------
    ValueError
        If the call breaks a descriptor limit, its destination rows overlap each
        other, or (unless ``allow_raw_pointers``) a pointer is not
        ``T.address_of(tensor[...])``, which the race checker needs to know
        which tensor a copy touches.
    """
    event = describe(call)
    if event.kind != "push":
        return event

    def err(message):
        raise ValueError(f"tirx.hexagon.dma_push: {message}")

    for name, access in (("src", event.reads), ("dst", event.writes)):
        if not _is_pointer(access.pointer.ty):
            err(f"{name} must be a pointer, such as T.address_of(V[0, 0]); got {access.pointer.ty}")
    if not allow_raw_pointers:
        for name, access in (("src", event.reads), ("dst", event.writes)):
            if access.base is None:
                err(f"{name} must be T.address_of(tensor[indices]), got {access.pointer}")

    _, _, dst_stride, src_stride, row_size, nrows = (_const_int(a) for a in call.args)
    for name, value in zip(PUSH_ARGS[2:], (dst_stride, src_stride, row_size, nrows)):
        if value is not None and value < 0:
            err(f"{name} must not be negative, got {value}")
    if nrows is not None and nrows > MAX_NROWS:
        err(f"nrows {nrows} exceeds the descriptor limit {MAX_NROWS}")
    if row_size is not None and row_size > MAX_ROW_SIZE:
        err(f"row_size {row_size} exceeds the descriptor limit {MAX_ROW_SIZE}")
    for name, value in (("dst_stride", dst_stride), ("src_stride", src_stride)):
        if value is not None and value > MAX_STRIDE:
            err(f"{name} {value} exceeds the descriptor limit {MAX_STRIDE}")
    if (
        nrows is not None
        and nrows > 1
        and row_size is not None
        and dst_stride is not None
        and dst_stride < row_size
    ):
        err(f"dst_stride {dst_stride} < row_size {row_size}: destination rows overlap")
    return event


def _is_pointer(ty) -> bool:
    if isinstance(ty, ir.PtrType):
        return True
    return isinstance(ty, ir.PrimType) and str(ty.dtype) == "handle"


def dma_calls(func) -> list:
    """Every DMA call in ``func``, in the order the IR visits them."""
    found = []

    def visit(call):
        if is_dma_call(call):
            found.append(call)
        return call

    tvm_ffi.structural_map(func, [(ir.Call, visit)])
    return found


def verify_function(func, *, allow_raw_pointers: bool = False) -> list:
    """Validate every DMA call in ``func``; return their events in IR order."""
    return [validate(c, allow_raw_pointers=allow_raw_pointers) for c in dma_calls(func)]
