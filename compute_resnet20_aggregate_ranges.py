#!/usr/bin/env python3
"""
Compute aggregate (category-wide) profiled ranges for ResNet-20 from the
full per-tensor profiled JSON, by grouping tensors by their embedded op
type (Conv, Relu, Add, Pad, Gemm) in the tensor name.

Given the scale (73 profiled tensors across ~21 Conv, ~19 Relu, ~9 Add
sites), this computes ONE global (widest) bound per category rather than
precise per-instance bounds -- trading some per-layer precision for
robustness and simplicity, following the same "start safe, refine via
real crashes" methodology used throughout this pipeline's development.

Usage:
    eval "$(python3 compute_resnet20_aggregate_ranges.py \
        resnet20_profiled_ranges.json --num-classes 10)"
"""

import argparse
import json
import re
import sys


CATEGORY_PATTERNS = {
    "RELU": re.compile(r"/Relu(_\d+)?_output_0$"),
    "CONV": re.compile(r"/Conv_output_0$"),
    "ADD": re.compile(r"/Add_output_0$"),
    "PAD": re.compile(r"/Pad(_\d+)?_output_0$"),
    "GAP": re.compile(r"/GlobalAveragePool_output_0$"),
}
LOGITS_PATTERN = re.compile(r"/Gemm_output_0$")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("profiled_json")
    parser.add_argument("--num-classes", type=int, required=True)
    args = parser.parse_args()

    with open(args.profiled_json) as f:
        ranges = json.load(f)

    category_min = {cat: None for cat in CATEGORY_PATTERNS}
    category_max = {cat: None for cat in CATEGORY_PATTERNS}
    logits_min, logits_max = None, None
    unmatched = []

    for name, r in ranges.items():
        if name in ("image", "prediction", "/Flatten_output_0"):
            continue  # handled separately / not annotation targets
        matched = False
        for cat, pattern in CATEGORY_PATTERNS.items():
            if pattern.search(name):
                matched = True
                if category_min[cat] is None or r["min"] < category_min[cat]:
                    category_min[cat] = r["min"]
                if category_max[cat] is None or r["max"] > category_max[cat]:
                    category_max[cat] = r["max"]
                break
        if matched:
            continue
        if LOGITS_PATTERN.search(name):
            if logits_min is None or r["min"] < logits_min:
                logits_min = r["min"]
            if logits_max is None or r["max"] > logits_max:
                logits_max = r["max"]
            continue
        unmatched.append(name)

    if unmatched:
        print(f"# WARNING: {len(unmatched)} tensor(s) did not match any "
              f"known category: {unmatched}", file=sys.stderr)

    for cat in CATEGORY_PATTERNS:
        if category_min[cat] is None:
            print(f"# WARNING: no tensors matched category {cat}", file=sys.stderr)
    if logits_min is None:
        print("# WARNING: no Gemm/logits tensor found", file=sys.stderr)

    print("# Aggregate (category-wide) profiled ranges:", file=sys.stderr)
    for cat in CATEGORY_PATTERNS:
        print(f"#   {cat}: [{category_min[cat]}, {category_max[cat]}]", file=sys.stderr)
    print(f"#   LOGITS: [{logits_min}, {logits_max}]", file=sys.stderr)

    # The single widest bound overall, used as the universal
    # --accumulator-range catch-all (matching the pattern already used
    # for the earlier models: the logits range is expected to be the
    # widest, same as it was for wider_mlp).
    all_mins = [v for v in category_min.values() if v is not None] + \
               ([logits_min] if logits_min is not None else [])
    all_maxs = [v for v in category_max.values() if v is not None] + \
               ([logits_max] if logits_max is not None else [])
    widest_min, widest_max = min(all_mins), max(all_maxs)

    exports = [
        ("RELU_MIN", category_min["RELU"]),
        ("RELU_MAX", category_max["RELU"]),
        ("CONV_MIN", category_min["CONV"]),
        ("CONV_MAX", category_max["CONV"]),
        ("ADD_MIN", category_min["ADD"]),
        ("ADD_MAX", category_max["ADD"]),
        ("PAD_MIN", category_min["PAD"]),
        ("PAD_MAX", category_max["PAD"]),
        ("GAP_MIN", category_min["GAP"]),
        ("GAP_MAX", category_max["GAP"]),
        ("LOGITS_MIN", logits_min),
        ("LOGITS_MAX", logits_max),
        ("ACCUMULATOR_MIN", widest_min),
        ("ACCUMULATOR_MAX", widest_max),
        ("SOFTMAX_EXP_MIN", 0.0),
        ("SOFTMAX_EXP_MAX", 1.0),
        ("SOFTMAX_SUM_MIN", 1.0),
        ("SOFTMAX_SUM_MAX", float(args.num_classes)),
    ]

    for name, value in exports:
        print(f"export {name}={value!r}")


if __name__ == "__main__":
    main()
