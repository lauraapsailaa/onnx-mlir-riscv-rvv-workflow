#!/usr/bin/env python3
"""
Load PROFILED (empirical) intermediate-layer ranges from profile_ranges.py's
JSON output, and emit them as bash `export VAR=value` statements -- the same
format compute_layer_ranges.py uses for its STATIC (interval-arithmetic)
ranges, so the two are interchangeable in the build script.

Usage:
    eval "$(python3 load_profiled_ranges.py mnist_profiled_ranges.json \
        --maxpool-tensor onnx::Reshape_5 \
        --relu-tensor onnx::Gemm_9 \
        --fc2-logits-tensor x.3)"
"""

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profiled_json")
    parser.add_argument("--maxpool-tensor", required=True,
                         help="ONNX tensor name for the MaxPool output")
    parser.add_argument("--relu-tensor", required=True,
                         help="ONNX tensor name for the ReLU output")
    parser.add_argument("--fc2-logits-tensor", required=True,
                         help="ONNX tensor name for fc2's logits "
                              "(Softmax's max-reduction input)")
    args = parser.parse_args()

    with open(args.profiled_json) as f:
        ranges = json.load(f)

    def get(name):
        if name not in ranges:
            print(f"ERROR: tensor '{name}' not found in {args.profiled_json}. "
                  f"Available tensors: {list(ranges.keys())}", file=sys.stderr)
            sys.exit(1)
        return ranges[name]["min"], ranges[name]["max"]

    maxpool_min, maxpool_max = get(args.maxpool_tensor)
    relu_min, relu_max = get(args.relu_tensor)
    fc2_min, fc2_max = get(args.fc2_logits_tensor)

    print(f"# Loaded PROFILED (empirical) ranges from {args.profiled_json}",
          file=sys.stderr)
    print(f"# MaxPool ({args.maxpool_tensor}): [{maxpool_min}, {maxpool_max}]",
          file=sys.stderr)
    print(f"# ReLU ({args.relu_tensor}):    [{relu_min}, {relu_max}]",
          file=sys.stderr)
    print(f"# fc2 logits ({args.fc2_logits_tensor}): [{fc2_min}, {fc2_max}]",
          file=sys.stderr)

    print(f"export MAXPOOL_MIN={maxpool_min!r}")
    print(f"export MAXPOOL_MAX={maxpool_max!r}")
    print(f"export RELU_MIN={relu_min!r}")
    print(f"export RELU_MAX={relu_max!r}")
    print(f"export SOFTMAX_MAX_MIN={fc2_min!r}")
    print(f"export SOFTMAX_MAX_MAX={fc2_max!r}")
    # Softmax's sum-of-exponentials range is structural, not input-dependent:
    # one term is always exactly exp(0)=1 (the max element itself), the
    # other 9 are each in [0,1], so the 10-term sum is always in [1,10] --
    # profiling doesn't change this, it's derived from the operation's own
    # structure, same as the exp range itself ([0,1], always, since Softmax
    # subtracts the max before exponentiating).
    print(f"export SOFTMAX_EXP_MIN=0.0")
    print(f"export SOFTMAX_EXP_MAX=1.0")


if __name__ == "__main__":
    main()
