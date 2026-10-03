#!/usr/bin/env python3
"""
Given a list of declaring loop line numbers (some of which may be nested
inside others, e.g. a conv layer's 3 nested affine.for reduction levels
all getting matched by patch_raised_casts.py's detection), keep only the
OUTERMOST ones -- excluding any line that falls within an earlier, kept
line's own body range.

Uses genuine nesting (brace-depth), not trip count: trip count alone is
unreliable here, since conv1's own outermost loop (3-channel RGB input)
shares trip count 3 with every OTHER conv layer's nested kh/kw levels.

Usage:
    python3 dedupe_nested_loops.py resnet20_after_raise.mlir 83 170 252 ...
    # prints only the outermost line numbers, one per line, in order
"""

import argparse
import sys


def compute_body_end(lines, decl_idx):
    """Return the 0-indexed line where this declaration's own body block
    closes, via brace depth."""
    depth = 0
    base_depth = None
    for j in range(decl_idx, len(lines)):
        depth += lines[j].count("{") - lines[j].count("}")
        if base_depth is None:
            base_depth = depth
        elif depth < base_depth:
            return j
    return len(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mlir_path")
    parser.add_argument("lines", nargs="+", type=int,
                         help="Declaring line numbers (1-indexed) to filter")
    args = parser.parse_args()

    with open(args.mlir_path) as f:
        lines = f.read().splitlines()

    # Sort by line number to process in file order.
    sorted_lines = sorted(args.lines)

    outermost = []
    covered_until = -1  # 0-indexed line up to which we're inside a kept loop's body
    for line_no in sorted_lines:
        idx = line_no - 1  # 0-indexed
        if idx <= covered_until:
            continue  # nested inside an already-kept outer loop
        outermost.append(line_no)
        body_end = compute_body_end(lines, idx)
        covered_until = body_end

    print(f"# Kept {len(outermost)} outermost line(s) out of "
          f"{len(sorted_lines)} given (excluded nested levels)",
          file=sys.stderr)
    for line_no in outermost:
        print(line_no)


if __name__ == "__main__":
    main()
