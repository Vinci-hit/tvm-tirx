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

"""Tests for gen_table.py on small inputs written in the format of LLVM's files."""

import tvm_tirx.testing
from tvm_tirx.backend.hexagon import hvx
from tvm_tirx.backend.hexagon.hvx import gen_table

TD = """
class Hexagon_v32i32_v32i32v32i32_Intrinsic<string GCCIntSuffix,
      list<IntrinsicProperty> intr_properties = [IntrNoMem]>
  : Hexagon_Intrinsic<GCCIntSuffix,
       [llvm_v32i32_ty], [llvm_v32i32_ty,llvm_v32i32_ty],
       intr_properties>;

class Hexagon_v32i32_v32i32v32i32i32_Intrinsic<string GCCIntSuffix,
      list<IntrinsicProperty> intr_properties = [IntrNoMem]>
  : Hexagon_Intrinsic<GCCIntSuffix,
       [llvm_v32i32_ty], [llvm_v32i32_ty,llvm_v32i32_ty,llvm_i32_ty],
       intr_properties>;

class Hexagon_v128i1_v32i32v32i32_Intrinsic<string GCCIntSuffix,
      list<IntrinsicProperty> intr_properties = [IntrNoMem]>
  : Hexagon_Intrinsic<GCCIntSuffix,
       [llvm_v128i1_ty], [llvm_v32i32_ty,llvm_v32i32_ty],
       intr_properties>;

class Hexagon__v128i1ptrv32i32_Intrinsic<string GCCIntSuffix,
      list<IntrinsicProperty> intr_properties = [IntrNoMem]>
  : Hexagon_Intrinsic<GCCIntSuffix,
       [], [llvm_v128i1_ty,llvm_ptr_ty,llvm_v32i32_ty],
       intr_properties>;

class Hexagon__ptri64_Intrinsic<string GCCIntSuffix,
      list<IntrinsicProperty> intr_properties = [IntrNoMem]>
  : Hexagon_Intrinsic<GCCIntSuffix,
       [], [llvm_ptr_ty,llvm_i64_ty],
       intr_properties>;

def int_hexagon_V6_vaddw_128B :
Hexagon_v32i32_v32i32v32i32_Intrinsic<"HEXAGON_V6_vaddw_128B">;

def int_hexagon_V6_vadd_hf_hf_128B :
Hexagon_v32i32_v32i32v32i32_Intrinsic<"HEXAGON_V6_vadd_hf_hf_128B">;

def int_hexagon_V6_valignbi_128B :
Hexagon_v32i32_v32i32v32i32i32_Intrinsic<"HEXAGON_V6_valignbi_128B",
  [IntrNoMem, ImmArg<ArgIndex<2>>]>;

def int_hexagon_V6_vgtw_128B :
Hexagon_v128i1_v32i32v32i32_Intrinsic<"HEXAGON_V6_vgtw_128B">;

def int_hexagon_V6_vS32b_qpred_ai_128B :
Hexagon__v128i1ptrv32i32_Intrinsic<"HEXAGON_V6_vS32b_qpred_ai_128B", [IntrWriteMem]>;

def int_hexagon_Y5_l2fetch :
Hexagon__ptri64_Intrinsic<"HEXAGON_Y5_l2fetch", []>;
"""


def block(syntax, proto, itype, slots, define, cond=None):
    lines = [f"#if {cond}"] if cond else []
    lines += [
        "/* " + "=" * 74,
        f"   Assembly Syntax:       {syntax}",
        f"   C Intrinsic Prototype: {proto}",
        f"   Instruction Type:      {itype}",
        f"   Execution Slots:       {slots}",
        "   " + "=" * 74 + " */",
        "",
        define,
    ]
    if cond:
        lines.append("#endif /* __HEXAGON_ARCH___ */")
    return "\n".join(lines) + "\n\n"


WRAP = "__BUILTIN_VECTOR_WRAP"
HVX_HEADER = (
    block(
        "Vd32.w=vadd(Vu32.w,Vv32.w)",
        "HVX_Vector Q6_Vw_vadd_VwVw(HVX_Vector Vu, HVX_Vector Vv)",
        "CVI_VA",
        "SLOT0123",
        f"#define Q6_Vw_vadd_VwVw(Vu,Vv) {WRAP}(__builtin_HEXAGON_V6_vaddw)(Vu,Vv)",
    )
    + block(
        "Vd32=valign(Vu32,Vv32,#u3)",
        "HVX_Vector Q6_V_valign_VVI(HVX_Vector Vu, HVX_Vector Vv, Word32 Iu3)",
        "CVI_VP",
        "SLOT0123",
        f"#define Q6_V_valign_VVI(Vu,Vv,Iu3) {WRAP}(__builtin_HEXAGON_V6_valignbi)(Vu,Vv,Iu3)",
    )
    + block(
        "Qd4=vcmp.gt(Vu32.w,Vv32.w)",
        "HVX_VectorPred Q6_Q_vcmp_gt_VwVw(HVX_Vector Vu, HVX_Vector Vv)",
        "CVI_VA",
        "SLOT0123",
        f"#define Q6_Q_vcmp_gt_VwVw(Vu,Vv) {WRAP}(__builtin_HEXAGON_V6_vandqrt)"
        f"(({WRAP}(__builtin_HEXAGON_V6_vgtw)(Vu,Vv)),-1)",
    )
    + block(
        "if (Qv4) vmem(Rt32+#s4)=Vs32",
        "void Q6_vmem_QRIV(HVX_VectorPred Qv, HVX_Vector* Rt, HVX_Vector Vs)",
        "CVI_VM_ST",
        "SLOT0",
        f"#define Q6_vmem_QRIV(Qv,Rt,Vs) {WRAP}(__builtin_HEXAGON_V6_vS32b_qpred_ai)"
        f"({WRAP}(__builtin_HEXAGON_V6_vandvrt)((Qv),-1),Rt,Vs)",
    )
    + block(
        "Vd32.hf=vadd(Vu32.hf,Vv32.hf)",
        "HVX_Vector Q6_Vhf_vadd_VhfVhf(HVX_Vector Vu, HVX_Vector Vv)",
        "CVI_VX",
        "SLOT23",
        f"#define Q6_Vhf_vadd_VhfVhf(Vu,Vv) {WRAP}(__builtin_HEXAGON_V6_vadd_hf_hf)(Vu,Vv)",
        cond="__HVX_ARCH__ >= 68 && defined __HVX_IEEE_FP__",
    )
    + block(
        "Vd32.w=vadd(Vu32.w,Vv32.w,Qx4):carry",
        "HVX_Vector Q6_Vw_vadd_VwVwQ_carry(HVX_Vector Vu, HVX_Vector Vv, HVX_VectorPred* Qx)",
        "CVI_VA",
        "SLOT0123",
        f"#define Q6_Vw_vadd_VwVwQ_carry(Vu,Vv,Qx) "
        f"{WRAP}(__builtin_HEXAGON_V6_vaddcarry)(Vu,Vv,Qx)",
        cond="__HVX_ARCH__ >= 62",
    )
)

SCALAR_HEADER = "".join(
    block(
        syntax,
        f"void Q6_{name}({params})",
        "ST",
        "SLOT0",
        f"#define Q6_{name} __builtin_HEXAGON_Y5_l2fetch",
    )
    for name, syntax, params in [
        ("l2fetch_AP", "l2fetch(Rs32,Rtt32)", "Address Rs, Word64 Rtt"),
    ]
)


def build(scalar_allowlist=("Q6_l2fetch_AP",)):
    original = gen_table.SCALAR_ALLOWLIST
    gen_table.SCALAR_ALLOWLIST = scalar_allowlist
    try:
        return gen_table.build_table(TD, HVX_HEADER, SCALAR_HEADER)
    finally:
        gen_table.SCALAR_ALLOWLIST = original


def test_generated_rows_match_the_shipped_table():
    entries, skipped = build()
    by_name = {e.name: e for e in hvx.ENTRIES}
    assert [e["name"] for e in entries] == [
        "Vw_vadd_VwVw",
        "V_valign_VVI",
        "Q_vcmp_gt_VwVw",
        "vmem_QRIV",
        "Vhf_vadd_VhfVhf",
        "l2fetch_AP",
    ]
    for e in entries:
        shipped = by_name[e["name"]]
        operands = tuple((o.name, o.llvm, o.imm) for o in shipped.operands)
        assert e["operands"] == operands
        for field in ["namespace", "intrinsic", "ret", "memory", "arch", "syntax", "itype"]:
            assert e[field] == getattr(shipped, field), (e["name"], field)
        assert (e["ieee_fp"], e["qfloat"]) == (shipped.ieee_fp, shipped.qfloat)
    assert skipped == [("Q6_Vw_vadd_VwVwQ_carry", "no LLVM intrinsic HEXAGON_V6_vaddcarry_128B")]


def test_predicate_conversions_are_stripped():
    builtin, args, returns_pred = gen_table.core_call(
        f"{WRAP}(__builtin_HEXAGON_V6_vandqrt)(({WRAP}(__builtin_HEXAGON_V6_vgtw_xor)"
        f"({WRAP}(__builtin_HEXAGON_V6_vandvrt)((Qx),-1),Vu,Vv)),-1)"
    )
    assert (builtin, args, returns_pred) == (
        "HEXAGON_V6_vgtw_xor",
        [("Qx", True), ("Vu", False), ("Vv", False)],
        True,
    )


def test_render_is_importable_python():
    entries, skipped = build()
    namespace = {}
    source = gen_table.render(entries, skipped, "0.0.0").replace(
        "from .schema import", "from tvm_tirx.backend.hexagon.hvx.schema import"
    )
    exec(compile(source, "table.py", "exec"), namespace)  # pylint: disable=exec-used
    assert namespace["LLVM_VERSION"] == "0.0.0"
    assert [e.name for e in namespace["ENTRIES"]] == [e["name"] for e in entries]
    assert namespace["ENTRIES"][1].operands[2].imm


if __name__ == "__main__":
    tvm_tirx.testing.main()
