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

"""Tests for the Hexagon DMA queue operations tirx.hexagon.dma_push / dma_pop / dma_flush."""

import re
from pathlib import Path

import pytest
import tvm_ffi

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx import ir
from tvm_tirx.backend.hexagon import dma
from tvm_tirx.script import tirx as T

HEXAGON_V79 = {
    "kind": "hexagon",
    "mtriple": "hexagon",
    "mcpu": "hexagonv73",  # LLVM 23 knows v79; v73 keeps the test runnable on older LLVM
    "mattr": ["+hvxv73", "+hvx-length128b"],
    "num-cores": 4,
    "vtcm-capacity": 8 * 2**20,
}


def has_hexagon_codegen():
    return tvm_ffi.get_global_func("target.build.hexagon", allow_missing=True) is not None


# fmt: off
@T.function
def add_one(A: T.Tensor((64, 128), "float16")):
    """DDR -> VTCM -> add 1 -> DDR, one 8-row tile at a time."""
    V = T.alloc_tensor(ty_args=[T.Tensor((8, 128), "float16", scope="global.vtcm")])
    for i in range(8):
        T.hexagon.dma_push(T.address_of(V[0, 0]), T.address_of(A[i * 8, 0]), 256, 256, 256, 8)
        T.hexagon.dma_pop()
        for j in range(8):
            for k in range(128):
                V[j, k] = V[j, k] + T.float16(1)
        T.hexagon.dma_push(T.address_of(A[i * 8, 0]), T.address_of(V[0, 0]), 256, 256, 256, 8)
        T.hexagon.dma_flush()
# fmt: on


def from_source(code):
    return tvm_tirx.script.from_source(
        code, extra_vars={"I": tvm_tirx.script.ir, "T": tvm_tirx.script.tirx}
    )


def test_ops_are_registered_as_opaque_void():
    from tvm_tirx.backend.hexagon import op as hexagon_op

    for name, op in [
        ("dma_push", dma.DMA_PUSH),
        ("dma_pop", dma.DMA_POP),
        ("dma_flush", dma.DMA_FLUSH),
    ]:
        assert op.name == f"tirx.hexagon.{name}"
        assert op.get_attr("TCallEffectKind") == ir.CallEffectKind.Opaque
    for call in [hexagon_op.dma_pop(), hexagon_op.dma_flush()]:
        assert str(call.ty) == str(ir.PrimType("void"))


def test_script_builds_statements_in_program_order():
    kinds = [dma.describe(c).kind for c in dma.dma_calls(add_one)]
    assert kinds == ["push", "pop", "push", "flush"]


def test_script_roundtrip():
    code = add_one.script()
    assert "T.hexagon.dma_push(T.address_of(V[0, 0]), T.address_of(A[i * 8, 0])" in code
    assert from_source(code).script() == code
    ir.assert_structural_equal(add_one, from_source(code))


def test_describe_resolves_tensors_and_scopes():
    load, _, store, _ = dma.verify_function(add_one)
    assert load.reads.base.tensor.name == "A" and load.reads.base.scope == "global"
    assert load.writes.base.tensor.name == "V" and load.writes.base.scope == "global.vtcm"
    assert store.reads.base.scope == "global.vtcm" and store.writes.base.scope == "global"
    assert [int(x) for x in (load.writes.row_size, load.writes.nrows, load.writes.stride)] == [
        256,
        8,
        256,
    ]
    assert not load.is_dummy


def _push_in_function(dst_stride, src_stride, row_size, nrows, raw=False):
    @T.function
    def f(A: T.Tensor((64, 128), "float16"), p: T.handle("float16")):
        V = T.alloc_tensor(ty_args=[T.Tensor((8, 128), "float16", scope="global.vtcm")])
        T.hexagon.dma_push(
            T.address_of(V[0, 0]),
            p if raw else T.address_of(A[0, 0]),
            dst_stride,
            src_stride,
            row_size,
            nrows,
        )
        T.hexagon.dma_pop()

    return f


def test_validate_accepts_dummy_push():
    (push, _) = dma.verify_function(_push_in_function(0, 0, 0, 0))
    assert push.is_dummy


@pytest.mark.parametrize(
    "args, message",
    [
        ((256, 256, 256, 70000), "nrows 70000 exceeds"),
        ((128, 256, 256, 8), "destination rows overlap"),
        ((256, 2**24, 256, 8), "src_stride 16777216 exceeds"),
        ((256, 256, -1, 8), "row_size must not be negative"),
    ],
)
def test_validate_rejects_bad_descriptors(args, message):
    with pytest.raises(ValueError, match=message):
        dma.verify_function(_push_in_function(*args))


def test_validate_requires_tensor_pointers_for_the_race_checker():
    f = _push_in_function(256, 256, 256, 8, raw=True)
    with pytest.raises(ValueError, match=r"src must be T\.address_of"):
        dma.verify_function(f)
    push, _ = dma.verify_function(f, allow_raw_pointers=True)
    assert push.reads.base is None


def test_type_checking_rejects_wrong_arity():
    with pytest.raises(Exception):
        ir.Call(dma.DMA_POP, [1]).validate()
    with pytest.raises(Exception):
        ir.Call(dma.DMA_PUSH, [1, 2, 3]).validate()


def _extern_calls(func):
    found = []

    def visit(call):
        if getattr(call.op, "name", "") == "tirx.call_extern":
            found.append((call.args[0].value, len(call.args) - 1))
        return call

    tvm_ffi.structural_map(func, [(ir.Call, visit)])
    return found


def test_lower_intrin_on_hexagon_calls_runtime_shim():
    func = add_one.with_attr("target", tvm_tirx.target.Target(HEXAGON_V79))
    lowered = tvm_tirx.tirx.transform.LowerIntrin()(ir.IRModule({"main": func}))["main"]
    assert not dma.dma_calls(lowered)
    assert _extern_calls(lowered) == [
        ("tirx_hexagon_dma_push", 6),
        ("tirx_hexagon_dma_pop", 0),
        ("tirx_hexagon_dma_push", 6),
        ("tirx_hexagon_dma_flush", 0),
    ]


def test_lower_intrin_leaves_other_targets_alone():
    func = add_one.with_attr("target", tvm_tirx.target.Target("llvm"))
    lowered = tvm_tirx.tirx.transform.LowerIntrin()(ir.IRModule({"main": func}))["main"]
    assert len(dma.dma_calls(lowered)) == 4


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


@pytest.mark.skipif(not has_hexagon_codegen(), reason="need TIRx built with LLVM")
def test_compile_emits_shim_calls_in_hexagon_code(fake_hexagon_linker):
    mod = ir.IRModule({"main": add_one.with_attr("tirx.noalias", True)})
    target = tvm_tirx.target.Target(HEXAGON_V79)
    built = tvm_tirx.compile(mod, target=tvm_tirx.target.Target(target, target))
    llvm_ir = built.mod.imports[0].inspect_source("ll")
    assert re.search(
        r"call void @tirx_hexagon_dma_push\(ptr .*, ptr .*, i32 256, i32 256, i32 256, i32 8\)",
        llvm_ir,
    ), llvm_ir
    assert "call void @tirx_hexagon_dma_pop()" in llvm_ir
    assert "call void @tirx_hexagon_dma_flush()" in llvm_ir
    asm = built.mod.imports[0].inspect_source("s")
    assert re.search(r"call tirx_hexagon_dma_push", asm), asm


if __name__ == "__main__":
    tvm_tirx.testing.main()
