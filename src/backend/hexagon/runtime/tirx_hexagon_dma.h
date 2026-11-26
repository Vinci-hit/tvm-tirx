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
 * \file tirx_hexagon_dma.h
 * \brief C runtime shim for the tirx.hexagon.dma_* operations.
 *
 * LowerIntrin turns tirx.hexagon.dma_push / dma_pop / dma_flush into calls to
 * the three functions below. They forward to the per-thread DMA queue of
 * llama.cpp (ggml/src/ggml-hexagon/htp/dma-queue.h), which the implementation
 * in tirx_hexagon_dma.c includes, so generated kernels and the reference
 * operators share one implementation of the queue.
 *
 * The queue is per thread and not named by the kernel. The code that launches
 * a kernel on a hardware thread binds that thread's queue first:
 *
 *   tirx_hexagon_dma_bind(ctx->dma[thread_index]);
 *   kernel(...);
 *   tirx_hexagon_dma_unbind();   // aborts if jobs are still outstanding
 *
 * Protocol errors (push to a full queue, pop from an empty queue, jobs left at
 * unbind) abort with a message, matching the protocol errors of the event
 * model in python/tvm_tirx/backend/hexagon/dma.py.
 *
 * Status: tirx_hexagon_dma.c builds only with the Hexagon SDK (dma-queue.h
 * needs HAP_farf.h and hexagon_types.h), so CMake does not build it; link it
 * into the DSP library next to llama.cpp's dma-queue.c.
 */
#ifndef TIRX_HEXAGON_DMA_H_
#define TIRX_HEXAGON_DMA_H_

#include <stdint.h>

struct dma_queue_s;

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Bind the calling thread's DMA queue before running a kernel on it. */
void tirx_hexagon_dma_bind(struct dma_queue_s* q);

/*! \brief Unbind it after the kernel; aborts if jobs are still outstanding. */
void tirx_hexagon_dma_unbind(void);

/*! \brief Lowered tirx.hexagon.dma_push. Aborts if the queue is full. */
void tirx_hexagon_dma_push(void* dst, const void* src, uint32_t dst_stride, uint32_t src_stride,
                           uint32_t row_size, uint32_t nrows);

/*! \brief Lowered tirx.hexagon.dma_pop. Aborts if the queue is empty. */
void tirx_hexagon_dma_pop(void);

/*! \brief Lowered tirx.hexagon.dma_flush. */
void tirx_hexagon_dma_flush(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TIRX_HEXAGON_DMA_H_
