# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""C source emission for statically shaped JAX kernels using MLIR EmitC."""

import logging
import shlex
import subprocess
from collections.abc import Sequence
from pathlib import Path

from jax.export import Exported
from jax.interpreters.mlir import make_ir_context
from jaxlib.mlir import ir, passmanager
from jaxlib.mlir.dialects import memref, stablehlo

logger = logging.getLogger(__name__)


DEFAULT_LINALG_PASSES = ("stablehlo-legalize-to-linalg",)
DEFAULT_LOOP_PASSES = (
    "inline",
    "func.func(linalg-detensorize{aggressive-mode})",
    "linalg-inline-scalar-operands",
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
    "mem2reg",
    "arith-expand",
    "canonicalize",
    "convert-math-to-emitc{language-target=c99}",
    "convert-to-emitc",
)


def _lower_to_linalg(source: str, context: ir.Context, name: str, passes: Sequence[str]) -> str:
    with context, ir.Location.unknown():
        module = ir.Module.parse(source)
        ir.SymbolTable(module.operation)["main"].attributes["sym_name"] = ir.StringAttr.get(name)
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


def _use_indexable_storage(op):
    # LLVM 22 EmitC needs indexable storage for surviving rank-zero stack slots.
    if op.name == "memref.alloca" and (scalar := ir.MemRefType(op.result.type)).rank == 0:
        op.result.set_type(
            ir.MemRefType.get([1], scalar.element_type, memory_space=scalar.memory_space)
        )
        view = memref.CollapseShapeOp(scalar, op.result, [], ip=ir.InsertionPoint.after(op))
        op.result.replace_all_uses_except(view.result, view.operation)
    return ir.WalkResult.ADVANCE


def _lowering_stages(exported, name, linalg_passes, loop_passes, emitc_passes):
    source = exported.mlir_module()
    yield "stablehlo.mlir", source

    context = make_ir_context()
    linalg = _lower_to_linalg(source, context, name, linalg_passes)
    yield "linalg.mlir", linalg

    loops = _mlir_opt(linalg, loop_passes)
    yield "loops.mlir", loops

    with context, ir.Location.unknown():
        module = ir.Module.parse(loops)
        module.operation.walk(_use_indexable_storage)
        shaped = str(module)
    yield "shaped.mlir", shaped
    emitc = _mlir_opt(shaped, emitc_passes)
    yield "emitc.mlir", emitc

    command = ("mlir-translate", "--mlir-to-cpp")
    logger.info("%s", shlex.join(command))
    c = subprocess.run(command, input=emitc, text=True, stdout=subprocess.PIPE, check=True).stdout
    source = (
        "#include <stdint.h>\n#include <stddef.h>\n#include <stdbool.h>\n#include <math.h>\n\n" + c
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
    *,
    name: str,
    output_dir: Path | None = None,
    linalg_passes: Sequence[str] = DEFAULT_LINALG_PASSES,
    loop_passes: Sequence[str] = DEFAULT_LOOP_PASSES,
    emitc_passes: Sequence[str] = DEFAULT_EMITC_PASSES,
) -> str:
    """Return C source for an exported, statically shaped JAX kernel with the given C name.

    Requires mlir-opt, mlir-translate and clang-format on PATH. Tested with LLVM 22.1.6.
    If supplied, output_dir receives the intermediate MLIR files and the generated C source.

    Each pass sequence replaces that stage's pipeline for this call. Entries use
    MLIR pass-pipeline syntax inside ``builtin.module``. For example, supply
    ``loop_passes=(*DEFAULT_LOOP_PASSES, "cse")`` to append a pass after loop lowering.
    """
    if output_dir is not None:
        output_dir = Path(output_dir)
        if Path(name).name != name:
            raise ValueError("The C function name must not contain a directory")
        output_dir.mkdir(exist_ok=True, parents=True)
    for filename, source in _lowering_stages(
        exported, name, linalg_passes, loop_passes, emitc_passes
    ):
        if output_dir is not None:
            (output_dir / filename).write_text(source)

    return source
