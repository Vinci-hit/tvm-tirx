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
"""Row types of the Hexagon instruction table.

Pure data: this module imports nothing from ``tvm_tirx``, so that
``gen_table.py`` and other tools can load the table standalone.

One :class:`Instruction` is one C intrinsic of the Hexagon programming
interface (the ``Q6_*`` functions of ``hvx_hexagon_protos.h`` and
``hexagon_protos.h``), bound to the LLVM intrinsic it expands to. Types are
LLVM IR types, spelled the way ``IntrinsicsHexagonDep.td`` spells them:

==========  ==============  ==========================================
LLVM type   C type          TIRx dtype
==========  ==============  ==========================================
``v32i32``  HVX_Vector      ``int32x32``, one 128-byte vector register
``v64i32``  HVX_VectorPair  ``int32x64``, a register pair
``v128i1``  HVX_VectorPred  ``boolx128``, a Q predicate register
``i32``     Word32          ``int32``
``i64``     Word64          ``int64``
``ptr``     Address         a pointer, such as ``T.address_of(A[i])``
==========  ==============  ==========================================
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Operand:
    """One operand, in the order of the C prototype (which is also LLVM's order)."""

    #: The parameter name of the C prototype, such as ``"Vu"``, ``"Rt"`` or ``"Iu3"``.
    name: str
    #: The LLVM IR type of the parameter.
    llvm: str
    #: Whether LLVM requires a compile-time constant here (``ImmArg``).
    imm: bool = False


@dataclass(frozen=True)
class Instruction:
    """One intrinsic: a ``Q6_*`` function and the LLVM intrinsic behind it."""

    #: ``"hvx"`` for vector instructions, ``"hexagon"`` for scalar memory-system ones.
    namespace: str
    #: The ``Q6_*`` name without the ``Q6_`` prefix, such as ``"Vw_vadd_VwVw"``.
    name: str
    #: The LLVM intrinsic, such as ``"llvm.hexagon.V6.vaddw.128B"``.
    intrinsic: str
    #: The LLVM return type, or ``""`` for ``void``.
    ret: str
    operands: tuple[Operand, ...]
    #: LLVM's memory model of the intrinsic: ``"none"`` (``IntrNoMem``),
    #: ``"write"`` (``IntrWriteMem``), ``"argmem"`` (``IntrArgMemOnly``), or
    #: ``"unknown"`` (no memory property: may read and write anything).
    memory: str
    #: The first HVX version that has the intrinsic, such as 60, 68 or 79.
    arch: int
    #: The assembly syntax, such as ``"Vd32.w=vadd(Vu32.w,Vv32.w)"``.
    syntax: str
    #: The instruction type, such as ``"CVI_VA"``.
    itype: str
    #: The execution slots, such as ``"SLOT0123"``.
    slots: str
    #: Needs the ``hvx-ieee-fp`` target feature (``__HVX_IEEE_FP__`` in the header).
    ieee_fp: bool = False
    #: Needs the ``hvx-qfloat`` target feature. Derived from the intrinsic name:
    #: every intrinsic that produces or consumes the qf16/qf32 formats has
    #: ``qf16`` or ``qf32`` in it.
    qfloat: bool = False

    @property
    def c_name(self) -> str:
        """The C function name, such as ``"Q6_Vw_vadd_VwVw"``."""
        return "Q6_" + self.name

    @property
    def op_name(self) -> str:
        """The TIRx operator name, such as ``"tirx.hvx.Vw_vadd_VwVw"``."""
        return f"tirx.{self.namespace}.{self.name}"
