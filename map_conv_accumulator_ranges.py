#!/usr/bin/env python3
"""
Map each Conv layer's own accumulator (the running sum during its
reduction loop, pre-bias) to a precise, per-layer bound, instead of the
single aggregate ACCUMULATOR_MIN/MAX bound used for all ~21 conv layers.

The profiled data we have (Conv_output_0) is POST-bias, but the
accumulator itself is PRE-bias. Since accumulator = conv_output - bias,
we compute: [conv_min - bias_max, conv_max - bias_min] (the worst-case
range of the accumulator given the known ranges of both). This doesn't
fully account for intermediate partial sums during summation potentially
overshooting the final value (if large positive and negative terms
partially cancel), so a multiplicative safety margin is added on top.

Conv_output_0 tensors (profiled JSON) and .bias entries (extract_ranges.py
JSON) are matched POSITIONALLY (both populated in graph-topological
order), NOT by name transformation -- the two files use different naming
conventions for the same layer (e.g. profiled
"/stage1/stage1.0/conv1/Conv_output_0" vs extracted
"stage1.0.conv1.bias") that aren't safely convertible via simple string
manipulation. A strict count-match check guards against misalignment.

Usage:
    python3 map_conv_accumulator_ranges.py \
        resnet20_profiled_ranges.json resnet20_ranges.json \
        --margin 1.5 \
        87 152 217 ... (CONV_ITERARG_LINES, in file order)
"""

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("profiled_json")
    parser.add_argument("extracted_json")
    parser.add_argument("--margin", type=float, default=1.5,
                         help="Multiplicative safety margin applied to the "
                              "bias-adjusted range, to account for potential "
                              "intermediate partial-sum overshoot during "
                              "summation that the post-bias profiled value "
                              "alone doesn't capture (default: 1.5x)")
    parser.add_argument("--flag", default="--iterarg-range",
                         help="Which flag to emit (--iterarg-range or --loop-range)")
    parser.add_argument("conv_lines", nargs="+", type=int,
                         help="Line numbers, in file order, for each conv layer's accumulator")
    args = parser.parse_args()

    with open(args.profiled_json) as f:
        profiled = json.load(f)
    with open(args.extracted_json) as f:
        extracted = json.load(f)

    conv_tensors = [(k, v) for k, v in profiled.items() if "/Conv_output_0" in k]
    bias_entries = [(k, v) for k, v in extracted.items()
                    if k.endswith(".bias") and v.get("role") == "weight"
                    and not k.startswith("fc.")]
    excluded = [k for k, v in extracted.items()
                if k.endswith(".bias") and v.get("role") == "weight"
                and k.startswith("fc.")]
    if excluded:
        print(f"# Excluded non-conv bias entries: {excluded}", file=sys.stderr)

    if len(conv_tensors) != len(bias_entries):
        print(f"ERROR: {len(conv_tensors)} profiled Conv_output_0 tensor(s) "
              f"vs {len(bias_entries)} .bias entries -- counts must match "
              f"before pairing positionally, refusing to guess.",
              file=sys.stderr)
        sys.exit(1)

    if len(args.conv_lines) != len(conv_tensors):
        print(f"ERROR: given {len(args.conv_lines)} line number(s) but "
              f"{len(conv_tensors)} Conv layer(s) found -- counts must "
              f"match exactly before pairing positionally, refusing to "
              f"guess.", file=sys.stderr)
        sys.exit(1)

    print(f"# Matched {len(args.conv_lines)} conv accumulator(s) to "
          f"bias-adjusted profiled ranges (margin={args.margin}x):",
          file=sys.stderr)
    flag_args = []
    for line_no, (conv_name, conv_info), (bias_name, bias_info) in zip(
            args.conv_lines, conv_tensors, bias_entries):
        conv_min, conv_max = conv_info["min"], conv_info["max"]
        bias_min, bias_max = bias_info["min"], bias_info["max"]

        # accumulator = conv_output - bias
        acc_min = conv_min - bias_max
        acc_max = conv_max - bias_min

        # Apply safety margin around the center.
        center = (acc_min + acc_max) / 2
        half_width = (acc_max - acc_min) / 2 * args.margin
        vmin = center - half_width
        vmax = center + half_width

        print(f"#   line {line_no} <-> {conv_name} (bias: {bias_name}): "
              f"conv=[{conv_min:.4f},{conv_max:.4f}] bias=[{bias_min:.4f},{bias_max:.4f}] "
              f"-> accumulator=[{vmin:.4f},{vmax:.4f}]", file=sys.stderr)
        flag_args.append(f"{args.flag} {line_no}:{vmin},{vmax}")

    print(" ".join(flag_args))


if __name__ == "__main__":
    main()