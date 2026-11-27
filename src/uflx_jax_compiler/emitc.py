# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""C source emission for statically shaped JAX kernels using MLIR EmitC."""

import logging
import re
import shlex
import subprocess
from collections.abc import Iterator, Sequence

from jax.export import Exported
from jax.interpreters.mlir import make_ir_context
from jaxlib.mlir import ir, passmanager
from jaxlib.mlir.dialects import stablehlo

logger = logging.getLogger(__name__)


DEFAULT_LINALG_PASSES = ("stablehlo-legalize-to-linalg",)
DEFAULT_LOOP_PASSES = (
    "inline",
    "linalg-inline-scalar-operands",
    "canonicalize",
    "cse",
    "func.func(linalg-fold-unit-extent-dims,linalg-fuse-elementwise-ops)",
    "canonicalize",
    "cse",
    (
        "one-shot-bufferize{bufferize-function-boundaries "
        "function-boundary-type-conversion=identity-layout-map buffer-alignment=0}"
    ),
    "convert-linalg-to-loops",
    "fold-memref-alias-ops",
    "lower-affine",
    "cse",
    "func.func(promote-buffers-to-stack{max-alloc-size-in-bytes=32768})",
)
DEFAULT_EMITC_PASSES = (
    "fold-memref-alias-ops",
    "canonicalize",
    "buffer-results-to-out-params{modify-public-functions hoist-static-allocs}",
    "buffer-deallocation-pipeline",
    "mem2reg",
    "canonicalize",
    "arith-expand",
    "convert-math-to-emitc{language-target=c99}",
    "convert-to-emitc",
)


def _lower_to_linalg(source: str, context: ir.Context, name: str, passes: Sequence[str]) -> str:
    with context, ir.Location.unknown():
        module = ir.Module.parse(source)
        entry = ir.SymbolTable(module.operation)["main"].opview
        entry.attributes["sym_name"] = ir.StringAttr.get(name)
        # One-Shot Bufferize may otherwise reuse a caller's input as writable storage.
        attrs = entry.arg_attrs if "arg_attrs" in entry.attributes else [{}] * len(entry.arguments)
        entry.arg_attrs = ir.ArrayAttr.get(
            [
                ir.DictAttr.get({**dict(arg), "bufferization.writable": ir.BoolAttr.get(False)})
                for arg in attrs
            ]
        )
        stablehlo.register_stablehlo_passes()
        pipeline = f"builtin.module({','.join(passes)})"
        logger.info("%s", pipeline)
        passmanager.PassManager.parse(pipeline).run(module.operation)
        return str(module)


def _mlir_opt(source: str, passes: Sequence[str]) -> str:
    command = ("mlir-opt", f"--pass-pipeline=builtin.module({','.join(passes)})")
    logger.info("%s", shlex.join(command))
    return subprocess.run(
        command, input=source, text=True, stdout=subprocess.PIPE, check=True
    ).stdout


def _require_mlir_tools():
    for tool in ("mlir-opt", "mlir-translate"):
        version = subprocess.check_output([tool, "--version"], text=True)
        match = re.search(r"LLVM version (\d+)", version)
        if not match or int(match[1]) < 23:
            raise RuntimeError(
                f"EmitC requires {tool} from LLVM 23 or newer. Found {version.strip()}"
            )


def lowering_stages(
    exported: Exported,
    name: str,
    *,
    linalg_passes: Sequence[str] = DEFAULT_LINALG_PASSES,
    loop_passes: Sequence[str] = DEFAULT_LOOP_PASSES,
    emitc_passes: Sequence[str] = DEFAULT_EMITC_PASSES,
) -> Iterator[tuple[str, str]]:
    """Yield (filename, source) pairs for each MLIR stage and the final C99 kernel.

    Requires mlir-opt and mlir-translate from LLVM 23 or newer, and clang-format on PATH.
    """
    _require_mlir_tools()
    source = exported.mlir_module()
    yield "stablehlo.mlir", source

    context = make_ir_context()
    linalg = _lower_to_linalg(source, context, name, linalg_passes)
    yield "linalg.mlir", linalg

    loops = _mlir_opt(linalg, loop_passes)
    yield "loops.mlir", loops

    emitc = _mlir_opt(loops, emitc_passes)
    yield "emitc.mlir", emitc

    command = ("mlir-translate", "--mlir-to-cpp")
    logger.info("%s", shlex.join(command))
    c = subprocess.run(command, input=emitc, text=True, stdout=subprocess.PIPE, check=True).stdout
    source = (
        "#include <stdint.h>\n#include <stddef.h>\n#include <stdbool.h>\n#include <math.h>\n"
        "#include <stdlib.h>\n#include <string.h>\n\n" + c
    )

    filename = f"{name}.c"
    command = ("clang-format", f"--assume-filename={filename}")
    logger.info("%s", shlex.join(command))
    formatted = subprocess.run(
        command, input=source, text=True, stdout=subprocess.PIPE, check=True
    ).stdout
    yield filename, formatted


def emit_c(
    exported: Exported,
    name: str,
    *,
    linalg_passes: Sequence[str] = DEFAULT_LINALG_PASSES,
    loop_passes: Sequence[str] = DEFAULT_LOOP_PASSES,
    emitc_passes: Sequence[str] = DEFAULT_EMITC_PASSES,
) -> str:
    """Return the formatted C99 source. Use lowering_stages for the intermediate MLIR."""
    stages = lowering_stages(
        exported,
        name,
        linalg_passes=linalg_passes,
        loop_passes=loop_passes,
        emitc_passes=emitc_passes,
    )
    return dict(stages)[f"{name}.c"]
