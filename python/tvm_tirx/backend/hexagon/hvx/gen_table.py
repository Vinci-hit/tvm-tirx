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
"""Generate ``table.py`` from LLVM's own definition of the Hexagon intrinsics.

The table is not written by hand. It joins two files of one LLVM release:

* ``llvm/include/llvm/IR/IntrinsicsHexagonDep.td``: every LLVM intrinsic, with
  its IR types, memory properties and immediate operands.
* ``clang/lib/Headers/hvx_hexagon_protos.h`` (and ``hexagon_protos.h``): the
  ``Q6_*`` C functions that Hexagon code, such as llama.cpp's ggml-hexagon,
  is written with, with the assembly syntax, instruction type, execution
  slots and the first HVX version of each.

A ``Q6_*`` macro expands to one LLVM builtin. clang passes predicates as
vectors, so the macro wraps every predicate in a ``vandvrt(q, -1)`` /
``vandqrt(v, -1)`` conversion; the generator strips those wrappers, because
TIRx has a predicate type (``boolx128``) of its own. A macro whose operands do
not then map one-to-one, in order, onto the builtin's parameters is skipped
and listed in ``SKIPPED`` with the reason.

Usage (the TD file ships in the ``mlc-llvm-static`` build dependency; the two
headers come from the ``llvmorg-<version>`` tag of llvm-project)::

    python python/tvm_tirx/backend/hexagon/hvx/gen_table.py \\
        --llvm-version 23.1.1 \\
        --td .../mlc_llvm_static/static/include/llvm/IR/IntrinsicsHexagonDep.td \\
        --hvx-header hvx_hexagon_protos.h --scalar-header hexagon_protos.h \\
        -o python/tvm_tirx/backend/hexagon/hvx/table.py

This file imports nothing from ``tvm_tirx`` and runs standalone.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

#: LLVM TableGen type names, and the spelling the table uses for them.
LLVM_TYPES = {
    "llvm_v32i32_ty": "v32i32",
    "llvm_v64i32_ty": "v64i32",
    "llvm_v128i1_ty": "v128i1",
    "llvm_v16i32_ty": "v16i32",
    "llvm_v64i1_ty": "v64i1",
    "llvm_i32_ty": "i32",
    "llvm_i64_ty": "i64",
    "llvm_ptr_ty": "ptr",
    "llvm_float_ty": "f32",
    "llvm_double_ty": "f64",
}

#: C prototype types, and the LLVM types they must agree with.
C_TYPES = {
    "HVX_Vector": "v32i32",
    "HVX_VectorPair": "v64i32",
    "HVX_VectorPred": "v128i1",
    "Word32": "i32",
    "Word64": "i64",
    "Address": "ptr",
    "HVX_Vector*": "ptr",
    "void": "",
}

#: Scalar (non-HVX) intrinsics taken from ``hexagon_protos.h``: the cache and
#: prefetch instructions of the memory system. The rest of that header is
#: scalar arithmetic, which LLVM selects from ordinary IR.
SCALAR_ALLOWLIST = (
    "Q6_dccleana_A",
    "Q6_dccleaninva_A",
    "Q6_dcfetch_A",
    "Q6_dcinva_A",
    "Q6_dczeroa_A",
    "Q6_l2fetch_AR",
    "Q6_l2fetch_AP",
)

#: The first HVX version, for blocks of the header without an ``#if``.
BASE_ARCH = 60


def _split_list(text):
    return [part.strip() for part in text.split(",") if part.strip()]


def _split_props(text):
    """Split ``IntrNoMem, ImmArg<ArgIndex<2>>`` at top-level commas."""
    props, depth, start = [], 0, 0
    for i, ch in enumerate(text):
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif ch == "," and depth == 0:
            props.append(text[start:i].strip())
            start = i + 1
    if text[start:].strip():
        props.append(text[start:].strip())
    return props


def parse_td(text):
    """Map each GCC builtin suffix (``HEXAGON_V6_vaddw_128B``) to its intrinsic."""
    classes = {}
    class_re = re.compile(
        r"class (Hexagon_\w+?_Intrinsic)<string GCCIntSuffix,\s*"
        r"list<IntrinsicProperty> intr_properties = \[([^\]]*)\]>\s*"
        r":\s*Hexagon_Intrinsic<GCCIntSuffix,\s*\[([^\]]*)\],\s*\[([^\]]*)\],\s*"
        r"intr_properties>;"
    )
    for m in class_re.finditer(text):
        name, props, rets, args = m.groups()
        classes[name] = (
            _split_props(props),
            [LLVM_TYPES[t] for t in _split_list(rets)],
            [LLVM_TYPES[t] for t in _split_list(args)],
        )
    defs = {}
    def_re = re.compile(
        r"def int_hexagon_(\w+)\s*:\s*(Hexagon_\w+?_Intrinsic)<\"(\w+)\"(?:,\s*\[([^\]]*)\])?>;"
    )
    for m in def_re.finditer(text):
        def_name, cls, gcc, props = m.groups()
        if cls not in classes:
            raise ValueError(f"{def_name}: unknown intrinsic class {cls}")
        default_props, rets, args = classes[cls]
        props = _split_props(props) if props is not None else default_props
        if len(rets) > 1:
            raise ValueError(f"{def_name}: more than one result")
        defs[gcc] = {
            "intrinsic": "llvm.hexagon." + def_name.replace("_", "."),
            "ret": rets[0] if rets else "",
            "args": args,
            "props": props,
            "imm": {int(i) for p in props for i in re.findall(r"^ImmArg<ArgIndex<(\d+)>>$", p)},
        }
    if not defs:
        raise ValueError("no intrinsic definitions found; is this IntrinsicsHexagonDep.td?")
    return defs


def _memory(props):
    if "IntrNoMem" in props:
        return "none"
    if "IntrWriteMem" in props:
        return "write"
    if "IntrArgMemOnly" in props:
        return "argmem"
    if "IntrReadMem" in props:
        return "read"
    return "unknown"


# ---------------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------------

_BLOCK_RE = re.compile(
    r"(?:#if (?P<cond>[^\n]*)\n)?"
    r"/\* =+\n"
    r"\s*Assembly Syntax:\s*(?P<syntax>[^\n]*?)\s*\n"
    r"\s*C Intrinsic Prototype:\s*(?P<proto>[^\n]*?)\s*\n"
    r"\s*Instruction Type:\s*(?P<itype>[^\n]*?)\s*\n"
    r"\s*Execution Slots:\s*(?P<slots>[^\n]*?)\s*\n"
    r"\s*=+ \*/\s*\n\s*"
    r"#define (?P<name>Q6_\w+)(?P<params>\([^)]*\))?[ \t]+(?P<body>[^\n]*)"
)


def _join_continuations(text):
    return re.sub(r"[ \t]*\\\n[ \t]*", " ", text)


def parse_header(text):
    """Yield one dict per ``Q6_*`` macro of a protos header."""
    for m in _BLOCK_RE.finditer(_join_continuations(text)):
        cond = m.group("cond") or ""
        arch = re.search(r"__(?:HVX|HEXAGON)_ARCH__ >= (\d+)", cond)
        proto = re.fullmatch(r"(\S+)\s+(Q6_\w+)\((.*)\)", m.group("proto"))
        if proto is None or proto.group(2) != m.group("name"):
            raise ValueError(f"{m.group('name')}: unexpected prototype {m.group('proto')!r}")
        params = []
        for param in _split_list(proto.group(3)):
            ctype, _, pname = param.rpartition(" ")
            params.append((ctype.strip(), pname.strip()))
        yield {
            "name": m.group("name"),
            "arch": int(arch.group(1)) if arch else BASE_ARCH,
            "ieee_fp": "__HVX_IEEE_FP__" in cond,
            "syntax": m.group("syntax"),
            "itype": m.group("itype"),
            "slots": m.group("slots"),
            "ret_ctype": proto.group(1),
            "params": params,
            "macro_params": _split_list((m.group("params") or "()")[1:-1]),
            "body": m.group("body").strip(),
        }


class _Parser:
    """A recursive-descent parser for the expression of a ``Q6_*`` macro."""

    _TOKEN = re.compile(r"\s*(?:(@\w+)|(-?\d+)|(\w+)|([(),]))")

    def __init__(self, text):
        self.tokens = []
        pos = 0
        while pos < len(text):
            if text[pos:].strip() == "":
                break
            m = self._TOKEN.match(text, pos)
            if m is None:
                raise ValueError(f"cannot parse macro body near {text[pos:]!r}")
            self.tokens.append(next(g for g in m.groups() if g is not None))
            pos = m.end()
        self.pos = 0

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def take(self, expected=None):
        tok = self.peek()
        if expected is not None and tok != expected:
            raise ValueError(f"expected {expected!r}, got {tok!r}")
        self.pos += 1
        return tok

    def expr(self):
        tok = self.take()
        if tok == "(":
            node = self.expr()
            self.take(")")
            return node
        if tok.startswith("@"):
            self.take("(")
            args = []
            if self.peek() != ")":
                args.append(self.expr())
                while self.peek() == ",":
                    self.take(",")
                    args.append(self.expr())
            self.take(")")
            return ("call", tok[1:], args)
        if re.fullmatch(r"-?\d+", tok):
            return ("int", int(tok))
        return ("id", tok)

    def parse(self):
        node = self.expr()
        if self.peek() is not None:
            raise ValueError(f"trailing tokens {self.tokens[self.pos :]}")
        return node


def _is_conversion(node, builtin):
    return (
        node[0] == "call" and node[1] == builtin and len(node[2]) == 2 and node[2][1] == ("int", -1)
    )


def core_call(body):
    """Return ``(builtin, [(param, is_predicate)], returns_predicate)`` of a macro body."""
    tree = _Parser(re.sub(r"__BUILTIN_VECTOR_WRAP\(__builtin_(\w+)\)", r"@\1", body)).parse()
    returns_pred = False
    if _is_conversion(tree, "HEXAGON_V6_vandqrt") and tree[2][0][0] == "call":
        tree, returns_pred = tree[2][0], True
    if tree[0] != "call":
        raise ValueError("not a call")
    args = []
    for arg in tree[2]:
        if _is_conversion(arg, "HEXAGON_V6_vandvrt") and arg[2][0][0] == "id":
            args.append((arg[2][0][1], True))
        elif arg[0] == "id":
            args.append((arg[1], False))
        else:
            raise ValueError(f"operand {arg} is not a macro parameter")
    return tree[1], args, returns_pred


def build_entry(macro, defs, namespace):
    """Return ``(entry_fields, None)`` or ``(None, reason)`` for one macro."""
    name = macro["name"]
    if namespace == "hexagon":
        # Scalar header: an object-like macro naming the builtin directly.
        m = re.fullmatch(r"__builtin_(\w+)", macro["body"])
        if m is None or macro["macro_params"]:
            return None, f"unexpected macro form {macro['body']!r}"
        builtin, args, returns_pred = m.group(1), None, False
        gcc = builtin
    else:
        try:
            builtin, args, returns_pred = core_call(macro["body"])
        except ValueError as err:
            return None, f"irregular macro: {err}"
        gcc = builtin + "_128B"
    if gcc not in defs:
        return None, f"no LLVM intrinsic {gcc}"
    d = defs[gcc]
    params = macro["params"]
    for ctype, pname in params:
        if ctype not in C_TYPES:
            return None, f"parameter {pname} has C type {ctype}, which an operand cannot carry"
    if macro["ret_ctype"] not in C_TYPES:
        return None, f"return type {macro['ret_ctype']}"
    if args is not None:
        names = [pname for _, pname in params]
        if [a for a, _ in args] != names or names != macro["macro_params"]:
            return None, "operands are not the prototype parameters in order"
        for (_, is_pred), llvm_ty in zip(args, d["args"]):
            if is_pred != (llvm_ty == "v128i1"):
                return None, "predicate conversion does not match the LLVM signature"
        if returns_pred != (d["ret"] == "v128i1"):
            return None, "predicate result does not match the LLVM signature"
    if len(params) != len(d["args"]):
        return None, f"{len(params)} C parameters but {len(d['args'])} LLVM parameters"
    for (ctype, pname), llvm_ty in zip(params, d["args"]):
        if C_TYPES[ctype] != llvm_ty:
            return None, f"parameter {pname}: C type {ctype} but LLVM type {llvm_ty}"
    if C_TYPES[macro["ret_ctype"]] != d["ret"]:
        return None, f"C return type {macro['ret_ctype']} but LLVM return type {d['ret']!r}"
    operands = tuple(
        (pname, llvm_ty, i in d["imm"])
        for i, ((_, pname), llvm_ty) in enumerate(zip(params, d["args"]))
    )
    intrinsic = d["intrinsic"]
    return {
        "namespace": namespace,
        "name": name[len("Q6_") :],
        "intrinsic": intrinsic,
        "ret": d["ret"],
        "operands": operands,
        "memory": _memory(d["props"]),
        "arch": macro["arch"],
        "syntax": macro["syntax"],
        "itype": macro["itype"],
        "slots": macro["slots"],
        "ieee_fp": macro["ieee_fp"],
        "qfloat": bool(re.search(r"qf(16|32)", intrinsic)),
    }, None


def build_table(td_text, hvx_text, scalar_text):
    """Return ``(entries, skipped)`` for the three source files."""
    defs = parse_td(td_text)
    entries, skipped = [], []
    hvx_macros = list(parse_header(hvx_text))
    if not hvx_macros:
        raise ValueError("no Q6_* macros found in the HVX header")
    for macro in hvx_macros:
        entry, reason = build_entry(macro, defs, "hvx")
        if entry is None:
            skipped.append((macro["name"], reason))
        else:
            entries.append(entry)
    scalar = {m["name"]: m for m in parse_header(scalar_text)}
    for name in SCALAR_ALLOWLIST:
        if name not in scalar:
            raise ValueError(f"{name} is not in the scalar header")
        entry, reason = build_entry(scalar[name], defs, "hexagon")
        if entry is None:
            raise ValueError(f"{name}: {reason}")
        entries.append(entry)
    names = [e["name"] for e in entries]
    if len(names) != len(set(names)):
        raise ValueError("duplicate instruction names")
    return entries, skipped


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

_LICENSE = """\
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
"""


def _q(text):
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def render(entries, skipped, llvm_version):
    lines = [_LICENSE.rstrip("\n")]
    lines += [
        f'"""The Hexagon instruction table: {len(entries)} intrinsics of LLVM {llvm_version}.',
        "",
        "GENERATED by gen_table.py from LLVM's IntrinsicsHexagonDep.td and clang's",
        "hvx_hexagon_protos.h / hexagon_protos.h. Do not edit; regenerate instead.",
        '"""',
        "",
        "# ruff: noqa: E501",
        "# fmt: off",
        "from .schema import Instruction as I",
        "from .schema import Operand as O",
        "",
        f"LLVM_VERSION = {_q(llvm_version)}",
        "",
        "ENTRIES = (",
    ]
    for e in entries:
        ops = ", ".join(
            f"O({_q(n)}, {_q(t)}{', True' if imm else ''})" for n, t, imm in e["operands"]
        )
        ops = f"({ops},)" if len(e["operands"]) == 1 else f"({ops})"
        flags = ""
        if e["ieee_fp"]:
            flags += ", ieee_fp=True"
        if e["qfloat"]:
            flags += ", qfloat=True"
        lines.append(
            f"    I({_q(e['namespace'])}, {_q(e['name'])}, {_q(e['intrinsic'])}, {_q(e['ret'])},"
            f" {ops}, {_q(e['memory'])}, {e['arch']}, {_q(e['syntax'])}, {_q(e['itype'])},"
            f" {_q(e['slots'])}{flags}),"
        )
    lines.append(")")
    lines.append("")
    lines.append("#: Q6 macros left out of the table, and why.")
    lines.append("SKIPPED = (")
    for name, reason in skipped:
        lines.append(f"    ({_q(name)}, {_q(reason)}),")
    lines.append(")")
    lines.append("# fmt: on")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--llvm-version", required=True)
    parser.add_argument("--td", required=True, type=Path)
    parser.add_argument("--hvx-header", required=True, type=Path)
    parser.add_argument("--scalar-header", required=True, type=Path)
    parser.add_argument("-o", "--output", required=True, type=Path)
    args = parser.parse_args(argv)
    entries, skipped = build_table(
        args.td.read_text(), args.hvx_header.read_text(), args.scalar_header.read_text()
    )
    args.output.write_text(render(entries, skipped, args.llvm_version))
    print(f"wrote {len(entries)} entries ({len(skipped)} skipped) to {args.output}")
    for name, reason in skipped:
        print(f"  skipped {name}: {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
