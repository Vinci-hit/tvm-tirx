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
 * \file src/backend/hexagon/op/dma.cc
 * \brief Hexagon per-thread DMA queue operations and their lowering.
 *
 * On the hexagon target, LowerIntrin turns each operation into a call to the
 * C runtime shim in src/backend/hexagon/runtime/tirx_hexagon_dma.h:
 *
 *   tirx.hexagon.dma_push(d, s, ds, ss, rs, n) -> tirx_hexagon_dma_push(d, s, ds, ss, rs, n)
 *   tirx.hexagon.dma_pop()                     -> tirx_hexagon_dma_pop()
 *   tirx.hexagon.dma_flush()                   -> tirx_hexagon_dma_flush()
 */
#include <tvm/backend/hexagon/op/dma.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/op.h>
#include <tvm/ir/prim/op.h>
#include <tvm/target/target.h>
#include <tvm/tirx/op/abi.h>
#include <tvm/tirx/op/attr.h>

#include <string>

namespace tvm {
namespace backend {
namespace hexagon {

using namespace tirx;

namespace {

/*! \brief Replace a DMA call with an impure extern call to the runtime shim. */
PrimExpr LowerToShim(const PrimExpr& e, const std::string& symbol) {
  const CallObj* call = e.as<CallObj>();
  TVM_FFI_ICHECK(call != nullptr);
  ffi::Array<Expr> args = {StringImm(symbol)};
  bool is_push = call->op.same_as(dma_push_op());
  int index = 0;
  for (const Expr& arg : call->args) {
    // The pointers pass through; the shim takes uint32 byte counts, as the
    // hardware descriptor does.
    if (is_push && index >= 2) {
      args.push_back(prim::cast(arg.as_or_throw<PrimExpr>(), PrimType::UInt(32)));
    } else {
      args.push_back(arg);
    }
    ++index;
  }
  return Call(PrimType::Void(), call_extern_op(), args, std::nullopt, ffi::Array<Type>(), call->loc)
      .as_or_throw<PrimExpr>();
}

}  // namespace

const Op& dma_push_op() {
  static const Op op = Op::Get("tirx.hexagon.dma_push");
  return op;
}

const Op& dma_pop_op() {
  static const Op op = Op::Get("tirx.hexagon.dma_pop");
  return op;
}

const Op& dma_flush_op() {
  static const Op op = Op::Get("tirx.hexagon.dma_flush");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.hexagon.dma_push",
        "Append an asynchronous 2D copy to the calling thread's DMA queue and return at once.")
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .signature(sig::arg("dst", "Destination pointer, T.address_of(tensor[indices])."),
                 sig::arg("src", "Source pointer, T.address_of(tensor[indices])."),
                 sig::arg<IntExpr>("dst_stride", "Bytes between consecutive destination rows."),
                 sig::arg<IntExpr>("src_stride", "Bytes between consecutive source rows."),
                 sig::arg<IntExpr>("row_size", "Bytes per row."),
                 sig::arg<IntExpr>("nrows", "Number of rows; 0 makes a dummy job."))
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque)
      .set_attr<FLowerIntrinsic>("hexagon.FLowerIntrinsic", [](PrimExpr e, Target) {
        return LowerToShim(e, "tirx_hexagon_dma_push");
      });

  OpDef("tirx.hexagon.dma_pop",
        "Wait for the oldest outstanding copy of the calling thread's DMA queue and remove it.")
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .signature()
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque)
      .set_attr<FLowerIntrinsic>("hexagon.FLowerIntrinsic", [](PrimExpr e, Target) {
        return LowerToShim(e, "tirx_hexagon_dma_pop");
      });

  OpDef("tirx.hexagon.dma_flush",
        "Wait for every outstanding copy of the calling thread's DMA queue and empty it.")
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .signature()
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque)
      .set_attr<FLowerIntrinsic>("hexagon.FLowerIntrinsic", [](PrimExpr e, Target) {
        return LowerToShim(e, "tirx_hexagon_dma_flush");
      });
}

}  // namespace hexagon
}  // namespace backend
}  // namespace tvm
