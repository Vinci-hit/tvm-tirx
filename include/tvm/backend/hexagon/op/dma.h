/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file tvm/backend/hexagon/op/dma.h
 * \brief Hexagon per-thread DMA queue operations.
 *
 * Each hardware thread owns one strict-FIFO DMA queue, as in the Hexagon
 * operators of llama.cpp (ggml/src/ggml-hexagon/htp/dma-queue.h). The
 * operations never name the queue: they act on the calling thread's queue.
 * See python/tvm_tirx/backend/hexagon/dma.py for the event model.
 */
#ifndef TVM_BACKEND_HEXAGON_OP_DMA_H_
#define TVM_BACKEND_HEXAGON_OP_DMA_H_

#include <tvm/ir/op.h>
#include <tvm/ir/prim/expr.h>

namespace tvm {
namespace backend {
namespace hexagon {

/*!
 * \brief Append an asynchronous 2D copy to the calling thread's DMA queue.
 *
 * Copies nrows rows of row_size bytes; row r goes from src + r * src_stride to
 * dst + r * dst_stride. nrows == 0 makes a dummy job that copies nothing but
 * must still be popped.
 *
 * Arguments, in order:
 * - args[0]: dst, Destination pointer.
 * - args[1]: src, Source pointer.
 * - args[2]: dst_stride, Bytes between consecutive destination rows.
 * - args[3]: src_stride, Bytes between consecutive source rows.
 * - args[4]: row_size, Bytes per row.
 * - args[5]: nrows, Number of rows.
 */
TVM_TIRX_DLL const Op& dma_push_op();

/*!
 * \brief Wait for the oldest outstanding copy of the calling thread's queue and remove it.
 */
TVM_TIRX_DLL const Op& dma_pop_op();

/*!
 * \brief Wait for every outstanding copy of the calling thread's queue and empty it.
 */
TVM_TIRX_DLL const Op& dma_flush_op();

}  // namespace hexagon
}  // namespace backend
}  // namespace tvm

#endif  // TVM_BACKEND_HEXAGON_OP_DMA_H_
