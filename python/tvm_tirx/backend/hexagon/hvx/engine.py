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
"""Generic engine for the table-driven Hexagon intrinsics.

One engine interprets every row of :mod:`.table`. No instruction has code of
its own:

- :func:`register_table` registers each row as the Op
  ``tirx.<namespace>.<name>``, such as ``tirx.hvx.Vw_vadd_VwVw``, with its
  effect kind, its result type and one shared ``hexagon`` lowering.
- :class:`HVXNamespace` builds calls by the row's C name without the ``Q6_``
  prefix, checking each operand against the C prototype. ``T.hvx`` holds the
  vector rows, and ``T.hexagon`` the scalar cache and prefetch rows.
- The lowering checks that the target has the HVX version and the features the
  row needs, then turns the call into a call of the row's LLVM intrinsic.

HVX registers are untyped in C: an ``HVX_Vector`` is 128 bytes whatever its
elements are. TIRx values have dtypes, so the engine maps types as follows.

- An ``HVX_Vector`` operand takes any 1024-bit vector, such as ``int8x128``,
  ``float16x64`` or ``int32x32``. An ``HVX_VectorPair`` operand takes any
  2048-bit vector, and an ``HVX_VectorPred`` operand takes ``boolx128``. The
  lowering bitcasts each operand to the type LLVM expects.
- The result dtype follows the prefix of the C name, which is the C API's own
  naming convention: ``Vhf_`` is ``float16x64``, ``Ww_`` is ``int32x64`` and
  ``Q_`` is ``boolx128``. qf16 and qf32 have no TIRx dtype, so their bits are
  ``int16`` and ``int32`` lanes.
- An untyped ``V_`` or ``W_`` result, such as that of ``V_vmux_QVV`` or
  ``V_hi_W``, keeps the element type of the first vector operand when the
  instruction only moves bits, which is when its assembly names no element
  type. Otherwise, or with no vector operand, it is ``int32`` lanes.

A widening instruction returns its pair in the order the hardware writes it,
which is usually even elements in the low vector and odd ones in the high
vector. The dtype says how many elements there are, not their order.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Iterable

from tvm_ffi import get_global_func

from tvm_tirx.ir import Call, CallEffectKind, IntImm, Op, const
from tvm_tirx.ir.op import register_op_attr
from tvm_tirx.ir.type import PrimType, PtrType
from tvm_tirx.script.ir_builder.base import auto_emit_void_call
from tvm_tirx.tirx.op import (
    call_llvm_intrin,
    call_llvm_pure_intrin,
    register_intrin_lowering,
    reinterpret,
)

from .schema import Instruction, Operand

#: The TIRx dtype of each LLVM type of the table, as LLVM sees it.
LLVM_DTYPES = {
    "v32i32": "int32x32",
    "v64i32": "int32x64",
    "v128i1": "boolx128",
    "i32": "int32",
    "i64": "int64",
    "": "void",
}

#: The C type of each LLVM type of the table, for messages and docs.
C_TYPES = {
    "v32i32": "HVX_Vector",
    "v64i32": "HVX_VectorPair",
    "v128i1": "HVX_VectorPred",
    "i32": "Word32",
    "i64": "Word64",
    "ptr": "Address",
    "": "void",
}

#: The element type of each type letter in a C name, as in ``Vhf_`` or ``Wuh_``.
ELEMENT_DTYPES = {
    "b": "int8",
    "ub": "uint8",
    "h": "int16",
    "uh": "uint16",
    "w": "int32",
    "uw": "uint32",
    "hf": "float16",
    "sf": "float32",
    "bf": "bfloat16",
    "qf16": "int16",
    "qf32": "int32",
}

_VECTOR_BITS = {"v32i32": 1024, "v64i32": 2048}
_SCALAR_BITS = {"i32": 32, "i64": 64}
_IMMEDIATE = re.compile(r"I([us])(\d+)")
_HVX_VERSION = re.compile(r"\+hvxv(\d+)")

# RemoveNoOp deletes an Evaluate whose call is at most kReadState, so every row
# that touches memory is kOpaque: a store, gather, scatter or cache operation
# must stay where it was written. A row that LLVM marks IntrNoMem is pure.
_EFFECT_PURE = CallEffectKind.Pure.value
_EFFECT_OPAQUE = CallEffectKind.Opaque.value

_ROWS: dict[str, Instruction] = {}
_BUILDERS: dict[str, object] = {}


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


def _vector_of(element: str, bits: int) -> str:
    return f"{element}x{bits // PrimType(element).dtype.bits}"


def result_dtype(entry: Instruction) -> str | None:
    """Return the dtype of a call to ``entry``, or None when it depends on the operands."""
    if entry.ret not in _VECTOR_BITS:
        return LLVM_DTYPES[entry.ret]
    element = entry.name.split("_", 1)[0][1:]
    if element:
        return _vector_of(ELEMENT_DTYPES[element], _VECTOR_BITS[entry.ret])
    if "." in entry.syntax or not any(o.llvm in _VECTOR_BITS for o in entry.operands):
        return _vector_of("int32", _VECTOR_BITS[entry.ret])
    return None


def _make_infer_type(entry: Instruction):
    """Type rule of an untyped row that only moves bits: keep the first vector's elements."""
    bits = _VECTOR_BITS[entry.ret]
    first = next(i for i, o in enumerate(entry.operands) if o.llvm in _VECTOR_BITS)
    width = _VECTOR_BITS[entry.operands[first].llvm]

    def infer_type(call):
        ty = call.args[first].ty if len(call.args) > first else None
        if isinstance(ty, PrimType) and _vector_bits(ty) == width:
            return PrimType(_vector_of(str(ty.dtype.with_lanes(1)), bits))
        return PrimType(_vector_of("int32", bits))

    return infer_type


def _vector_bits(ty: PrimType) -> int:
    """The bits of a non-bool vector dtype, or 0."""
    dtype = ty.dtype
    if dtype.lanes <= 1 or dtype.is_bool:
        return 0
    return dtype.bits * dtype.lanes


# ---------------------------------------------------------------------------
# Registration (import time)
# ---------------------------------------------------------------------------


def register_table(entries: Iterable[Instruction]) -> None:
    """Register every row as an Op with its effect, result type and lowering."""
    register_op = get_global_func("ir.RegisterOp")
    for entry in entries:
        name = entry.op_name
        register_op(name, describe(entry))
        Op.get(name).set_signature(args=[(o.name, C_TYPES[o.llvm]) for o in entry.operands])
        register_op_attr(
            name, "TCallEffectKind", _EFFECT_PURE if entry.memory == "none" else _EFFECT_OPAQUE
        )
        dtype = result_dtype(entry)
        if dtype is None:
            register_op_attr(name, "FInferType", _make_infer_type(entry))
        else:
            register_op_attr(name, "TFixedReturnType", PrimType(dtype))
        register_intrin_lowering(name, "hexagon", f=_lower)
        _ROWS[name] = entry


def lookup(namespace: str, name: str) -> Instruction:
    """Return the row of ``T.<namespace>.<name>``, such as ``lookup("hvx", "Vw_vadd_VwVw")``."""
    entry = _ROWS.get(f"tirx.{namespace}.{name}")
    if entry is None:
        raise KeyError(_unknown(namespace, name))
    return entry


def entries(namespace: str | None = None) -> list[Instruction]:
    """Return the registered rows, optionally of one namespace."""
    return [e for e in _ROWS.values() if namespace in (None, e.namespace)]


def describe(entry: Instruction) -> str:
    """The C prototype, assembly and scheduling facts of a row, as one docstring."""
    params = ", ".join(f"{C_TYPES[o.llvm]} {o.name}" for o in entry.operands)
    needs = [f"HVX v{entry.arch}"] if entry.namespace == "hvx" else []
    needs += ["hvx-ieee-fp"] * entry.ieee_fp + ["hvx-qfloat"] * entry.qfloat
    return (
        f"{C_TYPES[entry.ret]} {entry.c_name}({params})\n\n"
        f"Assembly: ``{entry.syntax}``\n"
        f"Type {entry.itype}, slots {entry.slots}"
        + (f", needs {', '.join(needs)}" if needs else "")
        + f".\nLLVM: ``{entry.intrinsic}``."
    )


def _unknown(namespace: str, name: str) -> str:
    message = f"T.{namespace} has no instruction {name!r}"
    if name.startswith("Q6_"):
        return f"{message}; drop the Q6_ prefix: T.{namespace}.{name[3:]}"
    for other in ("hvx", "hexagon"):
        if other != namespace and f"tirx.{other}.{name}" in _ROWS:
            return f"{message}; it is T.{other}.{name}"
    names = [e.name for e in entries(namespace)]
    close = difflib.get_close_matches(name, names, n=3)
    return message + (f"; did you mean {', '.join(close)}?" if close else "")


# ---------------------------------------------------------------------------
# Builders (T.hvx.*, and the table rows of T.hexagon.*)
# ---------------------------------------------------------------------------


def builder(namespace: str, name: str):
    """Return the builder of ``T.<namespace>.<name>``, or raise AttributeError."""
    key = f"tirx.{namespace}.{name}"
    found = _BUILDERS.get(key)
    if found is None:
        entry = _ROWS.get(key)
        if entry is None:
            raise AttributeError(_unknown(namespace, name))
        found = _BUILDERS[key] = _make_builder(entry)
    return found


def _make_builder(entry: Instruction):
    op = Op.get(entry.op_name)
    surface = f"T.{entry.namespace}.{entry.name}"

    def build(*args):
        if len(args) != len(entry.operands):
            names = ", ".join(o.name for o in entry.operands) or "none"
            raise TypeError(
                f"{surface} takes {len(entry.operands)} operands ({names}), got {len(args)}"
            )
        return Call(
            op,
            [
                _check_operand(surface, i, o, a)
                for i, (o, a) in enumerate(zip(entry.operands, args))
            ],
        )

    build.__name__ = build.__qualname__ = entry.name
    build.__doc__ = describe(entry)
    return auto_emit_void_call(build)


def _check_operand(surface: str, index: int, operand: Operand, value):
    """Return ``value`` as the Expr the operand takes, or raise TypeError or ValueError."""
    where = f"{surface}: operand {index} ({C_TYPES[operand.llvm]} {operand.name})"
    if operand.imm:
        if isinstance(value, IntImm):
            value = value.value
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{where} must be a compile-time integer, got {value!r}")
        signed, width = _IMMEDIATE.fullmatch(operand.name).groups()
        bits = int(width)
        low, high = (-(1 << (bits - 1)), 1 << (bits - 1)) if signed == "s" else (0, 1 << bits)
        if not low <= value < high:
            raise ValueError(f"{where} must be in [{low}, {high - 1}], got {value}")
        return const(value, "int32")
    if operand.llvm in _SCALAR_BITS:
        return _scalar(where, _SCALAR_BITS[operand.llvm], value)
    ty = getattr(value, "ty", None)
    if operand.llvm == "ptr":
        if not isinstance(ty, PtrType):
            raise TypeError(
                f"{where} must be a pointer, such as T.address_of(A[i]); got {_describe(value)}"
            )
        return value
    if isinstance(ty, PrimType):
        if operand.llvm == "v128i1":
            if ty.dtype.is_bool and ty.dtype.lanes == 128:
                return value
        elif _vector_bits(ty) == _VECTOR_BITS[operand.llvm]:
            return value
    wanted = {
        "v32i32": "a 1024-bit vector such as int32x32, int16x64 or float16x64",
        "v64i32": "a 2048-bit vector such as int32x64, int16x128 or float16x128",
        "v128i1": "a boolx128 predicate, such as the result of T.hvx.Q_vcmp_gt_VwVw",
    }[operand.llvm]
    raise TypeError(f"{where} must be {wanted}; got {_describe(value)}")


def _scalar(where: str, bits: int, value):
    if isinstance(value, bool) or not isinstance(value, int):
        ty = getattr(value, "ty", None)
        dtype = ty.dtype if isinstance(ty, PrimType) else None
        if dtype is not None and dtype.lanes == 1 and dtype.is_integer and dtype.bits == bits:
            return value
        raise TypeError(
            f"{where} must be an int{bits} or uint{bits} scalar; got {_describe(value)}"
        )
    if not -(1 << (bits - 1)) <= value < (1 << bits):
        raise ValueError(f"{where} does not fit in {bits} bits: {value}")
    # A Word32 literal may be written unsigned, as in C; keep its bits.
    return const(value - (1 << bits) if value >= (1 << (bits - 1)) else value, f"int{bits}")


def _describe(value) -> str:
    ty = getattr(value, "ty", None)
    return str(ty) if ty is not None else f"{type(value).__name__} {value!r}"


class HVXNamespace:
    """``T.hvx``: one builder per HVX intrinsic, named as in C without ``Q6_``.

    ``T.hvx.Vw_vadd_VwVw(a, b)`` builds the same instruction as C's
    ``Q6_Vw_vadd_VwVw(a, b)``. ``T.hvx["Q6_Vw_vadd_VwVw"]`` takes the C name.
    """

    def __init__(self, namespace: str = "hvx"):
        self._namespace = namespace

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return builder(self._namespace, name)

    def __getitem__(self, name: str):
        try:
            return builder(self._namespace, name.removeprefix("Q6_"))
        except AttributeError as err:
            raise KeyError(str(err)) from None

    def __dir__(self):
        return sorted({e.name for e in entries(self._namespace)} | set(super().__dir__()))

    def __repr__(self):
        return f"<T.{self._namespace}: {len(entries(self._namespace))} instructions>"


# ---------------------------------------------------------------------------
# Lowering (hexagon.FLowerIntrinsic)
# ---------------------------------------------------------------------------


def check_target(entry: Instruction, target) -> None:
    """Raise ValueError when ``target`` lacks the HVX version or a feature ``entry`` needs."""
    if entry.namespace != "hvx":
        return
    mattr = [str(m) for m in (target.attrs.get("mattr") or ())]
    versions = [int(m.group(1)) for m in map(_HVX_VERSION.fullmatch, mattr) if m]
    name = entry.c_name
    if not versions:
        raise ValueError(
            f"{name} is an HVX instruction, but the target enables no HVX version; "
            "add +hvxvNN to mattr, such as +hvxv79, or use a tag such as qcom/hexagon-v79"
        )
    if max(versions) < entry.arch:
        raise ValueError(f"{name} needs HVX v{entry.arch}, but the target enables v{max(versions)}")
    if "+hvx-length128b" not in mattr:
        raise ValueError(f"{name} is a 128-byte HVX instruction; add +hvx-length128b to mattr")
    if entry.ieee_fp and "+hvx-ieee-fp" not in mattr:
        raise ValueError(f"{name} is an IEEE floating-point instruction; add +hvx-ieee-fp to mattr")
    if entry.qfloat and "+hvx-qfloat" not in mattr:
        raise ValueError(f"{name} is a qfloat instruction; add +hvx-qfloat to mattr")


def _lower(call, target):
    entry = _ROWS[call.op.name]
    check_target(entry, target)
    args = []
    for arg, operand in zip(call.args, entry.operands):
        wanted = LLVM_DTYPES.get(operand.llvm)
        if wanted is not None and str(arg.ty.dtype) != wanted:
            arg = reinterpret(arg, ty=wanted, loc=call.loc)
        args.append(arg)
    llvm_dtype = LLVM_DTYPES[entry.ret]
    build = call_llvm_pure_intrin if entry.memory == "none" else call_llvm_intrin
    result = build(entry.intrinsic, *args, ty=llvm_dtype, loc=call.loc)
    if entry.ret and str(call.ty.dtype) != llvm_dtype:
        result = reinterpret(result, ty=call.ty, loc=call.loc)
    return result


__all__ = [
    "C_TYPES",
    "ELEMENT_DTYPES",
    "LLVM_DTYPES",
    "HVXNamespace",
    "builder",
    "check_target",
    "describe",
    "entries",
    "lookup",
    "register_table",
    "result_dtype",
]
