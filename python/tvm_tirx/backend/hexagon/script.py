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
"""Hexagon TVMScript namespace, available as ``T.hexagon``.

``T.hvx``, the HVX vector instructions, is :class:`~.hvx.HVXNamespace`.
"""

from __future__ import annotations

from tvm_tirx.backend.hexagon import op as _hexagon_op
from tvm_tirx.script.ir_builder.base import auto_emit_void_call as _auto_emit_void_call


class HexagonNamespace:
    """The Hexagon intrinsics submodule.

    It holds the DMA queue ops and, from the instruction table, the scalar cache
    and prefetch instructions, such as ``T.hexagon.l2fetch_AP``.
    """

    dma_push = staticmethod(_auto_emit_void_call(_hexagon_op.dma_push))
    dma_pop = staticmethod(_auto_emit_void_call(_hexagon_op.dma_pop))
    dma_flush = staticmethod(_auto_emit_void_call(_hexagon_op.dma_flush))

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        from .hvx import builder  # pylint: disable=import-outside-toplevel

        return builder("hexagon", name)

    def __dir__(self):
        from .hvx import entries  # pylint: disable=import-outside-toplevel

        return sorted({e.name for e in entries("hexagon")} | set(super().__dir__()))


__all__ = ["HexagonNamespace"]
