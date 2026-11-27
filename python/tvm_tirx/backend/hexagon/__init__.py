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
"""Hexagon-owned backend hooks."""

from importlib import import_module

_LAZY_SUBMODULES = {"dma", "hvx", "linker", "op", "script", "target_tags"}


def register_backend():
    """Register Hexagon-owned Python semantics."""
    from tvm_tirx.tirx.script.ir_builder import (
        op as builder_op,  # pylint: disable=import-outside-toplevel
    )

    import_module(f"{__name__}.target_tags")
    import_module(f"{__name__}.linker")
    for name, namespace in script_namespaces().items():
        builder_op.register_script_namespace(name, namespace)


def script_namespaces(**_):
    """Return Hexagon-owned TVMScript namespaces."""
    # pylint: disable=import-outside-toplevel
    from .hvx import HVXNamespace
    from .script import HexagonNamespace

    return {"hexagon": HexagonNamespace(), "hvx": HVXNamespace()}


def script_namespace(**kwargs):
    """Return the ``T.hexagon`` namespace object."""
    return script_namespaces(**kwargs)["hexagon"]


def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        return import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "dma",
    "hvx",
    "linker",
    "op",
    "register_backend",
    "script",
    "script_namespace",
    "script_namespaces",
    "target_tags",
]
