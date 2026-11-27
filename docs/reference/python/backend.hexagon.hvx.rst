..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

tvm_tirx.backend.hexagon.hvx
============================

.. automodule:: tvm_tirx.backend.hexagon.hvx
   :no-members:

.. currentmodule:: tvm_tirx.backend.hexagon.hvx

Classes
-------

.. autosummary::
   :toctree: generated/

   HVXNamespace

Functions
---------

.. autosummary::
   :toctree: generated/

   builder
   check_target
   describe
   entries
   lookup
   register_table
   result_dtype

Instruction table
-----------------

``python/tvm_tirx/backend/hexagon/hvx/table.py`` defines ``ENTRIES``, one
``Instruction`` record per C intrinsic. A record binds a ``Q6_*`` function of
the Hexagon programming interface to the LLVM intrinsic it expands to.

.. list-table::
   :header-rows: 1

   * - Field
     - Meaning
   * - ``namespace``, ``name``
     - The op ``tirx.<namespace>.<name>``, written ``T.<namespace>.<name>``.
       ``name`` is the C name without ``Q6_``.
   * - ``intrinsic``, ``ret``, ``operands``
     - The LLVM intrinsic and its IR types, in the order of the C prototype.
       An operand marked ``imm`` must be a compile-time integer.
   * - ``memory``
     - LLVM's memory model of the intrinsic. ``none`` makes the op pure;
       every other value makes it opaque, so that no pass removes or moves it.
   * - ``arch``, ``ieee_fp``, ``qfloat``
     - The first HVX version and the target features that the instruction
       needs. Lowering checks them against the target's ``mattr``.
   * - ``syntax``, ``itype``, ``slots``
     - The assembly syntax, the instruction type and the execution slots.

Types
-----

C passes HVX registers untyped. TIRx values carry dtypes, so the operands and
results map as follows.

.. list-table::
   :header-rows: 1

   * - C type
     - Operand takes
     - Result dtype
   * - ``HVX_Vector``
     - Any 1024-bit vector, such as ``int8x128`` or ``float16x64``
     - From the prefix of the C name: ``Vw_`` is ``int32x32``, ``Vhf_`` is
       ``float16x64``
   * - ``HVX_VectorPair``
     - Any 2048-bit vector
     - ``Ww_`` is ``int32x64``, ``Wh_`` is ``int16x128``
   * - ``HVX_VectorPred``
     - ``boolx128``
     - ``boolx128``
   * - ``Word32``, ``Word64``
     - A 32-bit or 64-bit integer scalar, or a Python ``int``
     - ``int32``
   * - ``Address``
     - A pointer, such as ``T.address_of(A[i])``
     - None

qf16 and qf32 have no TIRx dtype, so their bits are ``int16`` and ``int32``
lanes. An untyped ``V_`` or ``W_`` result, such as that of ``V_vmux_QVV``,
keeps the element type of the first vector operand when the instruction only
moves bits. Otherwise it is ``int32`` lanes.

.. code-block:: python

   a = A[T.Ramp(i * 64, 1, 64)]                       # float16x64
   b = B[T.Ramp(i * 64, 1, 64)]
   C[T.Ramp(i * 64, 1, 64)] = T.hvx.V_vmux_QVV(T.hvx.Q_vcmp_gt_VhfVhf(a, b), a, b)
   T.hexagon.l2fetch_AP(T.address_of(A[i * 64]), T.int64(0x0100008000000080))

Regenerating the table
----------------------

``gen_table.py`` joins two files of one LLVM release: the TableGen definitions
of the intrinsics, which ship in the ``mlc-llvm-static`` build dependency, and
clang's ``hvx_hexagon_protos.h`` and ``hexagon_protos.h`` from the
``llvmorg-<version>`` tag of llvm-project. It imports nothing from
``tvm_tirx``:

.. code-block:: bash

   python python/tvm_tirx/backend/hexagon/hvx/gen_table.py --llvm-version 23.1.1 \
       --td .../mlc_llvm_static/static/include/llvm/IR/IntrinsicsHexagonDep.td \
       --hvx-header hvx_hexagon_protos.h --scalar-header hexagon_protos.h \
       -o python/tvm_tirx/backend/hexagon/hvx/table.py
