#!/usr/bin/env python3
"""
Automatically classify every arith.maxnumf and math.exp occurrence in a
pre-raised MLIR file by semantic role, using structural heuristics --
replacing the manual grep+reasoning process used for the first two models
(MNIST, wider_mlp) with something reusable for any new model made of the
same building blocks (Gemm/ReLU/MaxPool/Softmax).

Classification rules:
  - arith.maxnumf where one operand is a zero constant (max(x, 0.0))
    -> "relu" (numbered in program order: relu_1, relu_2, ...)
  - arith.maxnumf where one operand is the ENCLOSING loop's own iter_arg
    -> "softmax_max" (Softmax's running-max reduction)
  - any other arith.maxnumf
    -> "maxpool" (or flagged "unclassified" if it doesn't fit the expected
       count, for manual review)
  - math.exp
    -> "softmax_exp" (always unambiguous)

Usage:
    python3 classify_ops.py wider_mlp_pre_taffo_fixed.mlir
"""

import argparse
import re
import sys


MAXNUMF_RE = re.compile(
    r'^(?P<indent>\s*)(?P<result>%\S+)\s*=\s*arith\.maxnumf\s+'
    r'(?P<lhs>%\S+),\s*(?P<rhs>%\S+)\s*:\s*f32'
)
MATH_EXP_RE = re.compile(
    r'^(?P<indent>\s*)(?P<result>%\S+)\s*=\s*math\.exp\s+%\S+\s*:\s*f32'
)
CONST_RE = re.compile(
    r'^\s*(?P<result>%\S+)\s*=\s*arith\.constant\s+(?P<value>\S+)\s*:\s*f32'
)
LOOP_DECL_RE = re.compile(
    r'^(?P<indent>\s*)(?P<result>%\S+)\s*=\s*(?:affine\.for|scf\.for)\s+.*'
    r'iter_args\((?P<itervar>%\S+)\s*=\s*%\S+\).*->\s*\(f32\)'
)

ZERO_LITERALS = {"0.000000e+00", "0.0", "0x00000000", "-0.000000e+00"}


def compute_brace_depths(lines):
    depths = []
    depth = 0
    for line in lines:
        depth += line.count("{") - line.count("}")
        depths.append(depth)
    return depths


def find_loop_iterargs(lines, depths):
    """Returns list of (itervar_name, loop_line, body_start_idx, body_end_idx)."""
    result = []
    for i, line in enumerate(lines):
        m = LOOP_DECL_RE.match(line)
        if m:
            base_depth = depths[i]
            body_end = len(lines)
            for j in range(i + 1, len(lines)):
                if depths[j] < base_depth:
                    body_end = j
                    break
            result.append((m.group("itervar"), i + 1, i, body_end))
    return result


def enclosing_iterargs(line_idx, loop_iterargs):
    """All iter_arg names of loops whose body encloses line_idx."""
    names = set()
    for itervar, loop_line, body_start, body_end in loop_iterargs:
        if body_start <= line_idx < body_end:
            names.add(itervar)
    return names


def is_zero_constant_operand(operand_name, line_idx, lines, window=2000):
    """
    Search backward from line_idx for the nearest definition of
    operand_name as a zero-valued arith.constant. Deliberately NOT
    depth-restricted: constants like the zero literal used by ReLU are
    typically hoisted and defined ONCE near the top of the function,
    outside any loop, then reused across many sibling loop bodies -- so
    stopping at the current block's boundary would miss them entirely.
    Exact SSA name matching is safe here since MLIR guarantees name
    uniqueness within a value's true dominating scope.
    """
    steps = 0
    for j in range(line_idx - 1, -1, -1):
        steps += 1
        if steps > window:
            break
        m = CONST_RE.match(lines[j])
        if m and m.group("result") == operand_name:
            return m.group("value") in ZERO_LITERALS
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mlir_path")
    args = parser.parse_args()

    with open(args.mlir_path) as f:
        lines = f.read().splitlines()

    depths = compute_brace_depths(lines)
    loop_iterargs = find_loop_iterargs(lines, depths)

    relu_count = 0
    results = []  # (line_no, category, detail)

    for i, line in enumerate(lines):
        m = MAXNUMF_RE.match(line)
        if m:
            lhs, rhs = m.group("lhs"), m.group("rhs")
            enclosing = enclosing_iterargs(i, loop_iterargs)
            if lhs in enclosing or rhs in enclosing:
                results.append((i + 1, "softmax_max", f"maxnumf {lhs}, {rhs}"))
            elif (is_zero_constant_operand(lhs, i, lines) or
                  is_zero_constant_operand(rhs, i, lines)):
                relu_count += 1
                results.append((i + 1, f"relu_{relu_count}", f"maxnumf {lhs}, {rhs}"))
            else:
                results.append((i + 1, "maxpool_or_unclassified", f"maxnumf {lhs}, {rhs}"))
            continue

        m = MATH_EXP_RE.match(line)
        if m:
            results.append((i + 1, "softmax_exp", "math.exp"))

    print(f"Classified {len(results)} op(s) in {args.mlir_path}:\n")
    for line_no, category, detail in results:
        print(f"  line {line_no:6d}  {category:28s} {detail}")

    print("\n--- Suggested --op-range / --iterarg-range entries ---")
    for line_no, category, detail in results:
        if category == "maxpool_or_unclassified":
            print(f"  line {line_no}: UNCLASSIFIED -- inspect manually "
                  f"(neither zero-constant nor loop-iterarg pattern matched; "
                  f"likely MaxPool, or a new pattern this script doesn't know about)")
        else:
            print(f"  line {line_no}: {category} -- assign the appropriate "
                  f"range for this role")


if __name__ == "__main__":
    main()
