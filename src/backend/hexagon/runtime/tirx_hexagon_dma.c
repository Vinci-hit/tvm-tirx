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
 * \file tirx_hexagon_dma.c
 * \brief Implementation of the tirx.hexagon.dma_* runtime shim over llama.cpp's dma_queue.
 */
#include "tirx_hexagon_dma.h"

#include <HAP_farf.h>
#include <stdlib.h>

#include "dma-queue.h"

/*! \brief The DMA queue of the calling hardware thread. */
static __thread dma_queue* tirx_hexagon_dma_q = NULL;

static void fail(const char* what) {
  FARF(ERROR, "tirx.hexagon.dma: %s", what);
  abort();
}

static dma_queue* current_queue(void) {
  if (tirx_hexagon_dma_q == NULL) fail("no DMA queue bound to this thread");
  return tirx_hexagon_dma_q;
}

void tirx_hexagon_dma_bind(struct dma_queue_s* q) { tirx_hexagon_dma_q = q; }

void tirx_hexagon_dma_unbind(void) {
  if (tirx_hexagon_dma_q != NULL && !dma_queue_empty(tirx_hexagon_dma_q)) {
    fail("jobs still outstanding when the kernel finished");
  }
  tirx_hexagon_dma_q = NULL;
}

void tirx_hexagon_dma_push(void* dst, const void* src, uint32_t dst_stride, uint32_t src_stride,
                           uint32_t row_size, uint32_t nrows) {
  if (!dma_queue_push(current_queue(), dma_make_data(dst, src), dst_stride, src_stride, row_size,
                      nrows)) {
    fail("push to a full queue");
  }
}

void tirx_hexagon_dma_pop(void) {
  dma_queue* q = current_queue();
  if (dma_queue_empty(q)) fail("pop from an empty queue");
  (void)dma_queue_pop(q);
}

void tirx_hexagon_dma_flush(void) { dma_queue_flush(current_queue()); }
