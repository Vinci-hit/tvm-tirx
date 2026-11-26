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
"""Hexagon-owned TIR intrinsic builders: the per-thread DMA queue.

See :mod:`tvm_tirx.backend.hexagon.dma` for the semantics and event model.
"""

from __future__ import annotations

from tvm_tirx.ir import Op
from tvm_tirx.ir._expose_api import _make_op_api

dma_push = _make_op_api(Op.get("tirx.hexagon.dma_push"), __name__)
dma_pop = _make_op_api(Op.get("tirx.hexagon.dma_pop"), __name__)
dma_flush = _make_op_api(Op.get("tirx.hexagon.dma_flush"), __name__)

__all__ = ["dma_flush", "dma_pop", "dma_push"]
