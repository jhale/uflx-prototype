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

from jax.interpreters import mlir
from jax.stages import Lowered
from jaxlib.mlir import ir, passmanager
from jaxlib.mlir.dialects import memref, stablehlo

logger = logging.getLogger(__name__)


DEFAULT_LINALG_PASSES = ("stablehlo-legalize-to-linalg",)
DEFAULT_LOOP_PASSES = (
    "inline",
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
    # Give rank-zero storage an array representation for EmitC. Alias folding
    # turns accesses through the scalar view into accesses at index zero.
    if op.name == "memref.global":
        scalar = ir.MemRefType(op.attributes["type"].value)
        if scalar.rank == 0:
            array = ir.MemRefType.get([1], scalar.element_type, memory_space=scalar.memory_space)
            op.attributes["type"] = ir.TypeAttr.get(array)
            initial = op.opview.initial_value
            if isinstance(initial, ir.DenseElementsAttr):
                tensor = ir.RankedTensorType.get([1], scalar.element_type)
                op.attributes["initial_value"] = ir.DenseElementsAttr.get_splat(
                    tensor, initial.get_splat_value()
                )
    elif op.name in ("memref.alloca", "memref.get_global") and (
        scalar := ir.MemRefType(op.result.type)
    ).rank == 0:
        op.result.set_type(
            ir.MemRefType.get([1], scalar.element_type, memory_space=scalar.memory_space)
        )
        view = memref.CollapseShapeOp(scalar, op.result, [], ip=ir.InsertionPoint.after(op))
        op.result.replace_all_uses_except(view.result, view.operation)
    return ir.WalkResult.ADVANCE


def _lowering_stages(lowered, name, linalg_passes, loop_passes, emitc_passes):
    stable = lowered.compiler_ir("stablehlo")
    # JAX may lift large closed-over tables into leading arguments. Embed them
    # in a copy so the C function takes only the kernel's runtime inputs.
    with stable.context, ir.Location.unknown():
        module = ir.Module.parse(str(stable))
        main = ir.SymbolTable(module.operation)["main"].opview
        attrs = list(main.arg_attrs) if "arg_attrs" in main.attributes else []
        captured = [i for i, attr in enumerate(attrs) if "jax.const" in attr]
        if captured:
            block = main.entry_block
            with ir.InsertionPoint.at_block_begin(block):
                for i, value in zip(captured, lowered._lowering.const_args, strict=True):
                    block.arguments[i].replace_all_uses_with(mlir.ir_constant(value))
            for i in reversed(captured):
                block.erase_argument(i)
                del attrs[i]
            main.function_type = ir.TypeAttr.get(
                ir.FunctionType.get([arg.type for arg in block.arguments], main.type.results)
            )
            main.arg_attrs = ir.ArrayAttr.get(attrs)
        source = str(module)
    yield "stablehlo.mlir", source

    linalg = _lower_to_linalg(source, stable.context, name, linalg_passes)
    yield "linalg.mlir", linalg

    loops = _mlir_opt(linalg, loop_passes)
    yield "loops.mlir", loops

    with stable.context, ir.Location.unknown():
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
    lowered: Lowered,
    *,
    name: str,
    output_dir: Path | None = None,
    linalg_passes: Sequence[str] = DEFAULT_LINALG_PASSES,
    loop_passes: Sequence[str] = DEFAULT_LOOP_PASSES,
    emitc_passes: Sequence[str] = DEFAULT_EMITC_PASSES,
) -> str:
    """Return C source for a lowered, statically shaped JAX kernel with the given C name.

    Requires mlir-opt, mlir-translate and clang-format on PATH. Tested with LLVM 22.1.6.
    If supplied, output_dir receives the intermediate MLIR files and the generated C source.

    Each pass sequence replaces that stage's pipeline for this call. Entries use
    MLIR pass-pipeline syntax inside builtin.module. For example, supply
    loop_passes=(*DEFAULT_LOOP_PASSES, "cse") to append a pass after loop lowering.
    """
    if output_dir is not None:
        output_dir = Path(output_dir)
        if Path(name).name != name:
            raise ValueError("The C function name must not contain a directory")
        output_dir.mkdir(exist_ok=True, parents=True)
    for filename, source in _lowering_stages(
        lowered, name, linalg_passes, loop_passes, emitc_passes
    ):
        if output_dir is not None:
            (output_dir / filename).write_text(source)

    return source
