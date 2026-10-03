#!/usr/bin/env python3
"""
Find every %allocN buffer that's read via an unannotated
"builtin.unrealized_conversion_cast %X : f32 to !taffo.real" (the marker
for a value TAFFO's range analysis will crash on), and show the context of
whatever STORES into that buffer -- enough to quickly judge its category
(ReLU output, raw Conv output, skip-connection Add result, GlobalAveragePool
output, etc.) without a full gdb round-trip for each one.

This does NOT decide the category for you -- that judgment call stays
manual on purpose, since a wrong guess would silently produce a
plausible-but-wrong numeric result, worse than a crash. It just surfaces
the relevant context fast.

Usage:
    python3 find_unannotated_buffers.py resnet20_after_raise_patched.mlir
    python3 find_unannotated_buffers.py resnet20_after_raise.mlir --context 8
"""

import argparse
import re
import sys
from collections import defaultdict


LOAD_CAST_RE = re.compile(
    r'^\s*(%\S+)\s*=\s*affine\.load\s+(%alloc(?:_\d+)?)\[[^\]]*\]\s*:\s*memref<'
)
UNANNOTATED_CAST_RE = re.compile(
    r'^\s*%\S+\s*=\s*builtin\.unrealized_conversion_cast\s+(%\S+)\s*:\s*f32\s+to\s+!taffo\.real\s*$'
)
STORE_RE = re.compile(
    r'^\s*affine\.store\s+(%\S+),\s*(%alloc(?:_\d+)?)\['
)


def find_unannotated_buffers(lines, window=5):
    """
    For each unannotated cast, look backward up to `window` lines for the
    nearest preceding "affine.load %allocN[...]" whose result matches the
    cast's operand. Matching via a nearest-preceding-line search (rather
    than a global name->buffer dict) avoids a real scope-collision bug:
    SSA names like %45 are reused constantly across unrelated scopes, so
    a global dict would let a later, unrelated load silently overwrite an
    earlier mapping and misattribute a completely different cast to the
    wrong buffer (the same bug class found and fixed in
    patch_raised_casts.py's find_loop_results).
    """
    buffers = set()
    for i, line in enumerate(lines):
        m = UNANNOTATED_CAST_RE.match(line)
        if not m:
            continue
        operand = m.group(1)
        for j in range(i - 1, max(-1, i - 1 - window), -1):
            lm = LOAD_CAST_RE.match(lines[j])
            if lm and lm.group(1) == operand:
                buffers.add(lm.group(2))
                break
    return buffers


def find_stores(lines, buffer_name):
    """Return list of (line_idx, stored_value_name) for every affine.store
    into this specific buffer."""
    result = []
    for i, line in enumerate(lines):
        m = STORE_RE.match(line)
        if m and m.group(2) == buffer_name:
            result.append((i, m.group(1)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mlir_path")
    parser.add_argument("--context", type=int, default=6,
                         help="Lines of context to show before each store (default: 6)")
    args = parser.parse_args()

    with open(args.mlir_path) as f:
        lines = f.read().splitlines()

    buffers = find_unannotated_buffers(lines)

    if not buffers:
        print("No unannotated buffers found.", file=sys.stderr)
        return

    print(f"Found {len(buffers)} unannotated buffer(s):\n", file=sys.stderr)

    def sort_key(b):
        parts = b.split("_")
        return int(parts[1]) if len(parts) > 1 else -1  # bare "%alloc" sorts first

    for buf in sorted(buffers, key=sort_key):
        stores = find_stores(lines, buf)
        print(f"=== {buf} ===", file=sys.stderr)
        if not stores:
            print("  (no affine.store found writing to this buffer -- "
                  "check manually, may be a function argument or global)",
                  file=sys.stderr)
        for line_idx, stored_val in stores:
            start = max(0, line_idx - args.context)
            print(f"  Store at line {line_idx + 1} (stores {stored_val}):",
                  file=sys.stderr)
            for j in range(start, line_idx + 1):
                marker = ">>" if j == line_idx else "  "
                print(f"    {marker} {j+1}: {lines[j].strip()}", file=sys.stderr)
            print("", file=sys.stderr)
        print("", file=sys.stderr)


if __name__ == "__main__":
    main()