#!/usr/bin/env python3
"""
Map a list of %allocN buffer names (in file order, as already discovered
via find_unannotated_buffers.py + manual/pattern-based classification) to
their own precise profiled ranges, instead of a single shared aggregate
bound -- matching tensor names containing a given substring (e.g. "/Pad"),
in graph-topological (profiled JSON insertion) order.

Relies on file order matching profiled-JSON (graph topological) order,
which is validated by requiring an EXACT count match before pairing --
if the counts don't match, this refuses to guess and exits with an error.

Usage:
    python3 map_buffer_ranges.py resnet20_profiled_ranges.json \
        --pattern /Pad \
        %alloc %alloc_4 %alloc_7 ... (buffer names, in file order)
    # prints --buffer-range %allocN:MIN,MAX for each, in order
"""

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ranges_json")
    parser.add_argument("--pattern", required=True,
                         help="Substring to match tensor names against, e.g. '/Pad'")
    parser.add_argument("buffers", nargs="+",
                         help="Buffer names (e.g. %%alloc_4), in file order")
    args = parser.parse_args()

    with open(args.ranges_json) as f:
        ranges = json.load(f)

    matched_tensors = [(k, v) for k, v in ranges.items() if args.pattern in k]
    # Rely on Python dict preserving insertion order, which
    # profile_ranges.py populates in graph-topological order.

    if len(args.buffers) != len(matched_tensors):
        print(f"ERROR: given {len(args.buffers)} buffer name(s) but "
              f"{len(matched_tensors)} tensor(s) matching pattern "
              f"{args.pattern!r} in the profiled JSON -- counts must "
              f"match exactly before pairing positionally, refusing to "
              f"guess.", file=sys.stderr)
        sys.exit(1)

    print(f"# Matched {len(args.buffers)} buffer(s) to profiled tensors "
          f"matching {args.pattern!r} (file order <-> graph order):",
          file=sys.stderr)
    buffer_range_args = []
    for buf_name, (tensor_name, info) in zip(args.buffers, matched_tensors):
        vmin, vmax = info["min"], info["max"]
        print(f"#   {buf_name} <-> {tensor_name}: [{vmin}, {vmax}]",
              file=sys.stderr)
        buffer_range_args.append(f"--buffer-range {buf_name}:{vmin},{vmax}")

    print(" ".join(buffer_range_args))


if __name__ == "__main__":
    main()
