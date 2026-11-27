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

"""Tests for the table-driven Hexagon intrinsics: T.hvx and the scalar T.hexagon rows."""

import re
from pathlib import Path

import pytest
import tvm_ffi

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx import ir
from tvm_tirx.backend.hexagon import hvx
from tvm_tirx.script import tirx as T

V79 = tvm_tirx.target.Target("qcom/hexagon-v79")


def has_hexagon_codegen():
    return tvm_ffi.get_global_func("target.build.hexagon", allow_missing=True) is not None


def hexagon(*mattr):
    return tvm_tirx.target.Target(
        {"kind": "hexagon", "mtriple": "hexagon", "mcpu": "hexagonv79", "mattr": list(mattr)}
    )


def from_source(code):
    return tvm_tirx.script.from_source(
        code, extra_vars={"I": tvm_tirx.script.ir, "T": tvm_tirx.script.tirx}
    )


def vec(dtype):
    return tvm_tirx.tirx.Var("v", dtype)


# fmt: off
@T.function
def kernel(A: T.Tensor((256,), "int32"), B: T.Tensor((256,), "int32"),
           C: T.Tensor((256,), "int32"), X: T.Tensor((128,), "float32"),
           Y: T.Tensor((128,), "float32"), Z: T.Tensor((128,), "float32"), n: T.int32):
    """Word add with a masked tail store, a byte rotate, and an fp32 multiply through qf32."""
    for i in range(8):
        T.hexagon.l2fetch_AP(T.address_of(A[i * 32]), T.int64(0x0100008000000080))
        a = A[T.Ramp(i * 32, 1, 32)]
        b = B[T.Ramp(i * 32, 1, 32)]
        s = T.hvx.Vw_vadd_VwVw(a, b)
        T.hvx.vmem_QRIV(T.hvx.Q_vsetq_R(n * 4), T.address_of(C[i * 32]), s)
        B[T.Ramp(i * 32, 1, 32)] = T.hvx.V_valign_VVI(s, a, 4)
    for i in range(4):
        x = X[T.Ramp(i * 32, 1, 32)]
        y = Y[T.Ramp(i * 32, 1, 32)]
        Z[T.Ramp(i * 32, 1, 32)] = T.hvx.Vsf_equals_Vqf32(T.hvx.Vqf32_vmpy_VsfVsf(x, y))
    T.hexagon.dccleaninva_A(T.address_of(Z[0]))


@T.function
def half_max(A: T.Tensor((128,), "float16"), B: T.Tensor((128,), "float16"),
             C: T.Tensor((128,), "float16")):
    for i in range(2):
        a = A[T.Ramp(i * 64, 1, 64)]
        b = B[T.Ramp(i * 64, 1, 64)]
        C[T.Ramp(i * 64, 1, 64)] = T.hvx.V_vmux_QVV(T.hvx.Q_vcmp_gt_VhfVhf(a, b), a, b)
# fmt: on


# ---------------------------------------------------------------------------
# The table and its registration
# ---------------------------------------------------------------------------


def test_every_row_is_registered():
    assert len(hvx.entries()) == len(hvx.ENTRIES)
    assert {e.namespace for e in hvx.ENTRIES} == {"hvx", "hexagon"}
    for entry in hvx.ENTRIES:
        op = ir.Op.get(entry.op_name)
        assert [a.name for a in op.args_info] == [o.name for o in entry.operands]
        assert op.get_attr("hexagon.FLowerIntrinsic") is not None


def test_table_rows_are_consistent():
    for entry in hvx.ENTRIES:
        prefix = entry.name.split("_", 1)[0]
        if entry.ret == "v32i32":
            assert prefix[0] == "V", entry
        elif entry.ret == "v64i32":
            assert prefix[0] == "W", entry
        elif entry.ret == "v128i1":
            assert prefix[0] == "Q", entry
        assert entry.memory in {"none", "write", "argmem", "unknown"}, entry
        assert (entry.memory == "none") == (entry.ret != ""), entry
        for operand in entry.operands:
            assert operand.imm == bool(re.fullmatch(r"I[us]\d+", operand.name)), entry
        hvx.result_dtype(entry)  # every prefix names a known element type


@pytest.mark.skipif(not has_hexagon_codegen(), reason="need TIRx built with LLVM")
def test_every_intrinsic_exists_in_the_linked_llvm():
    lookup = tvm_tirx.codegen.llvm.lookup_intrinsic_id
    assert [e.intrinsic for e in hvx.ENTRIES if lookup(e.intrinsic) == 0] == []


@pytest.mark.parametrize(
    "name, effect",
    [
        ("tirx.hvx.Vw_vadd_VwVw", "Pure"),
        ("tirx.hvx.Q_vcmp_gt_VwVw", "Pure"),
        ("tirx.hvx.vmem_QRIV", "Opaque"),
        ("tirx.hvx.vgather_ARMVh", "Opaque"),
        ("tirx.hvx.vscatter_RMVhV", "Opaque"),
        ("tirx.hexagon.l2fetch_AP", "Opaque"),
        ("tirx.hexagon.dccleaninva_A", "Opaque"),
    ],
)
def test_effect_kind_follows_llvm_memory_model(name, effect):
    assert ir.Op.get(name).get_attr("TCallEffectKind") == getattr(ir.CallEffectKind, effect)


def test_remove_no_op_keeps_memory_ops_and_drops_unused_pure_ones():
    # fmt: off
    @T.function
    def func(A: T.Tensor((32,), "int32")):
        T.evaluate(T.hvx.V_vzero())
        T.hexagon.l2fetch_AR(T.address_of(A[0]), 128)
    # fmt: on

    body = tvm_tirx.tirx.transform.RemoveNoOp()(ir.IRModule({"main": func}))["main"].script()
    assert "T.hvx.V_vzero" not in body
    assert "T.hexagon.l2fetch_AR(T.address_of(A[0]), 128)" in body


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, args, dtype",
    [
        ("Vw_vadd_VwVw", ["int32x32", "int32x32"], "int32x32"),
        ("Vhf_vadd_VhfVhf", ["float16x64", "float16x64"], "float16x64"),
        ("Vuh_vabsdiff_VhVh", ["int16x64", "int16x64"], "uint16x64"),
        ("Wh_vadd_VubVub", ["uint8x128", "uint8x128"], "int16x128"),
        ("Vqf32_vmpy_VsfVsf", ["float32x32", "float32x32"], "int32x32"),
        ("Q_vcmp_gt_VwVw", ["int32x32", "int32x32"], "boolx128"),
        ("R_vextract_VR", ["int8x128", "int32"], "int32"),
        # Untyped results that only move bits keep the first vector's elements.
        ("V_vmux_QVV", ["boolx128", "float16x64", "float16x64"], "float16x64"),
        ("V_hi_W", ["float16x128"], "float16x64"),
        ("W_vcombine_VV", ["uint8x128", "uint8x128"], "uint8x256"),
        ("V_vdelta_VV", ["float16x64", "uint8x128"], "float16x64"),
        # With no vector operand, or element types in the assembly, they are int32 lanes.
        ("V_vsplat_R", ["int32"], "int32x32"),
        ("V_vcvt_VhfVhf", ["float16x64", "float16x64"], "int32x32"),
    ],
)
def test_result_dtype(name, args, dtype):
    call = getattr(T.hvx, name)(*[vec(a) for a in args])
    assert str(call.ty.dtype) == dtype


def test_vector_operands_take_any_dtype_of_the_register_width():
    for dtype in ["int8x128", "uint16x64", "float16x64", "bfloat16x64", "float32x32"]:
        assert str(T.hvx.V_vand_VV(vec(dtype), vec("int32x32")).ty.dtype) == dtype


@pytest.mark.parametrize(
    "name, args, error, match",
    [
        ("Vw_vadd_VwVw", ["int32x16", "int32x32"], TypeError, r"operand 0 .*1024-bit vector"),
        ("Vw_vadd_VwVw", ["int32x32"], TypeError, r"takes 2 operands \(Vu, Vv\), got 1"),
        ("Vw_vadd_VwVw", ["boolx128", "int32x32"], TypeError, r"1024-bit vector"),
        ("Vw_vadd_VwVw", [1, 2], TypeError, r"1024-bit vector"),
        ("W_vcombine_VV", ["int32x64", "int32x32"], TypeError, r"1024-bit vector"),
        ("V_hi_W", ["int32x32"], TypeError, r"2048-bit vector"),
        ("V_vmux_QVV", ["boolx64", "int32x32", "int32x32"], TypeError, r"boolx128 predicate"),
        ("V_vsplat_R", ["float32"], TypeError, r"int32 or uint32 scalar"),
        ("V_vsplat_R", [1 << 33], ValueError, r"does not fit in 32 bits"),
        ("V_valign_VVI", ["int32x32", "int32x32", 8], ValueError, r"in \[0, 7\], got 8"),
        ("V_valign_VVI", ["int32x32", "int32x32", "int32"], TypeError, r"compile-time integer"),
        ("vmem_QRIV", ["boolx128", "int32", "int32x32"], TypeError, r"must be a pointer"),
    ],
)
def test_operand_errors(name, args, error, match):
    args = [vec(a) if isinstance(a, str) else a for a in args]
    with pytest.raises(error, match=match):
        getattr(T.hvx, name)(*args)


def test_word_literals_keep_their_bits():
    assert T.hvx.V_vsplat_R(0xFFFFFFFF).args[0].value == -1
    assert str(T.hvx.V_vsplat_R(T.uint32(7)).args[0].ty) == str(ir.PrimType("uint32"))


def test_unknown_names_point_to_the_right_one():
    with pytest.raises(AttributeError, match=r"drop the Q6_ prefix: T.hvx.Vw_vadd_VwVw"):
        T.hvx.Q6_Vw_vadd_VwVw
    with pytest.raises(AttributeError, match=r"it is T.hexagon.l2fetch_AP"):
        T.hvx.l2fetch_AP
    with pytest.raises(AttributeError, match=r"did you mean Vw_vadd_VwVw"):
        T.hvx.Vw_vadd_VwVx
    assert T.hvx["Q6_Vw_vadd_VwVw"] is T.hvx.Vw_vadd_VwVw
    assert "Vw_vadd_VwVw" in dir(T.hvx) and "l2fetch_AP" in dir(T.hexagon)


def test_builder_doc_is_the_c_prototype():
    doc = T.hvx.Vw_vadd_VwVw.__doc__
    assert "HVX_Vector Q6_Vw_vadd_VwVw(HVX_Vector Vu, HVX_Vector Vv)" in doc
    assert "Vd32.w=vadd(Vu32.w,Vv32.w)" in doc and "llvm.hexagon.V6.vaddw.128B" in doc


# ---------------------------------------------------------------------------
# Script
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("func", [kernel, half_max])
def test_script_roundtrip(func):
    code = func.script()
    assert from_source(code).script() == code
    ir.assert_structural_equal(func, from_source(code))


def test_script_prints_typed_values_and_statements():
    code = kernel.script()
    assert "s: T.let[T.int32x32] = T.hvx.Vw_vadd_VwVw(a, b)" in code
    assert "T.hvx.vmem_QRIV(T.hvx.Q_vsetq_R(n * 4), T.address_of(C[i * 32]), s)" in code
    assert "T.hvx.V_valign_VVI(s, a, 4)" in code
    assert "T.hexagon.dccleaninva_A(T.address_of(Z[0]))" in code


# ---------------------------------------------------------------------------
# Lowering
# ---------------------------------------------------------------------------


def lower(func, target):
    mod = ir.IRModule({"main": func.with_attr("target", target)})
    return tvm_tirx.tirx.transform.LowerIntrin()(mod)["main"].script()


def test_lower_intrin_calls_the_llvm_intrinsics():
    code = lower(kernel, V79)
    assert "T.hvx." not in code and "T.hexagon." not in code
    assert re.search(
        r'T\.call_llvm_pure_intrin\("llvm\.hexagon\.V6\.vaddw\.128B", a, b, ty=T\.int32x32\)', code
    ), code
    assert 'T.call_llvm_intrin("llvm.hexagon.V6.vS32b.qpred.ai.128B"' in code
    assert 'T.call_llvm_intrin("llvm.hexagon.Y5.l2fetch"' in code
    # Typed operands and results are bitcast to and from LLVM's v32i32.
    code = lower(half_max, V79)
    assert "T.reinterpret(a, ty=T.int32x32)" in code
    assert re.search(r"T\.reinterpret\(T\.call_llvm_pure_intrin\(.*vmux", code), code


def test_lower_intrin_leaves_other_targets_alone():
    assert "T.hvx.Vw_vadd_VwVw" in lower(kernel, tvm_tirx.target.Target("llvm"))


@pytest.mark.parametrize(
    "target, match",
    [
        (hexagon("+hvx-length128b"), r"enables no HVX version; add \+hvxvNN"),
        (
            hexagon("+hvxv66", "+hvx-length128b"),
            r"Q6_\w+ needs HVX v68, but the target enables v66",
        ),
        (hexagon("+hvxv79", "+hvx-length128b"), r"Q6_\w*qf32\w* is a qfloat instruction"),
        (hexagon("+hvxv79", "+hvx-qfloat"), r"128-byte HVX instruction; add \+hvx-length128b"),
        (tvm_tirx.target.Target("qcom/hexagon-v75"), None),
    ],
)
def test_lower_intrin_checks_the_target(target, match):
    if match is None:
        lower(kernel, target)
        return
    with pytest.raises(ValueError, match=match):
        lower(kernel, target)


def test_lower_intrin_checks_ieee_fp():
    with pytest.raises(ValueError, match=r"Q6_Vhf_vadd_VhfVhf is an IEEE floating-point"):
        lower(_ieee_add, tvm_tirx.target.Target("qcom/hexagon-v75"))
    lower(_ieee_add, V79)


# fmt: off
@T.function
def _ieee_add(A: T.Tensor((64,), "float16")):
    A[T.Ramp(0, 1, 64)] = T.hvx.Vhf_vadd_VhfVhf(A[T.Ramp(0, 1, 64)], A[T.Ramp(0, 1, 64)])
# fmt: on


# ---------------------------------------------------------------------------
# Hexagon code
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_hexagon_linker():
    name = "tvm.contrib.hexagon.link_shared"
    original = tvm_ffi.get_global_func(name)

    def link_shared(so_name, _objs, _extra_args=None):
        Path(so_name).write_bytes(b"")
        return 0

    tvm_ffi.register_global_func(name, link_shared, override=True)
    yield
    tvm_ffi.register_global_func(name, original, override=True)


def test_v79_tag():
    attrs = V79.attrs
    assert attrs["mcpu"] == "hexagonv79"
    assert list(attrs["mattr"]) == ["+hvxv79", "+hvx-length128b", "+hvx-qfloat", "+hvx-ieee-fp"]


@pytest.mark.skipif(not has_hexagon_codegen(), reason="need TIRx built with LLVM")
def test_compile_emits_hvx_instructions(fake_hexagon_linker):
    mod = ir.IRModule({"main": kernel.with_attr("tirx.noalias", True)})
    built = tvm_tirx.compile(mod, target=tvm_tirx.target.Target(V79, V79))
    llvm_ir = built.mod.imports[0].inspect_source("ll")
    for call in [
        r"call <32 x i32> @llvm\.hexagon\.V6\.vaddw\.128B\(<32 x i32> .*, <32 x i32> .*\)",
        r"call <128 x i1> @llvm\.hexagon\.V6\.pred\.scalar2\.128B\(i32 .*\)",
        r"call void @llvm\.hexagon\.V6\.vS32b\.qpred\.ai\.128B\(<128 x i1> .*, ptr .*, <32 x i32> ",
        r"call <32 x i32> @llvm\.hexagon\.V6\.valignbi\.128B\(<32 x i32> .*, i32 4\)",
        r"call <32 x i32> @llvm\.hexagon\.V6\.vmpy\.qf32\.sf\.128B",
        r"call <32 x i32> @llvm\.hexagon\.V6\.vconv\.sf\.qf32\.128B",
        r"call void @llvm\.hexagon\.Y5\.l2fetch\(ptr .*, i64 72058143793741952\)",
        r"call void @llvm\.hexagon\.Y2\.dccleaninva\(ptr .*\)",
    ]:
        assert re.search(call, llvm_ir), call
    asm = built.mod.imports[0].inspect_source("s")
    for instr in [
        r"v\d+\.w = vadd\(v\d+\.w,v\d+\.w\)",
        r"q\d+ = vsetq\(r\d+\)",
        r"if \(q\d+\) vmem\(r\d+\+#\d+\) = v\d+",
        r"v\d+ = valign\(v\d+,v\d+,#4\)",
        r"v\d+\.qf32 = vmpy\(v\d+\.sf,v\d+\.sf\)",
        r"v\d+\.sf = v\d+\.qf32",
        r"l2fetch\(r\d+,r\d+:\d+\)",
        r"dccleaninva\(r\d+\)",
    ]:
        assert re.search(instr, asm), instr


if __name__ == "__main__":
    tvm_tirx.testing.main()
