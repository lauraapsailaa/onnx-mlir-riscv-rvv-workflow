#!/usr/bin/env python3
"""
Load PROFILED (empirical) intermediate-layer ranges from profile_ranges.py's
JSON output, for an ARBITRARY chain of ReLU layers, and emit them as bash
`export VAR=value` statements -- matching compute_layer_ranges_n.py's output
variable naming (RELU1_MIN/MAX, RELU2_MIN/MAX, ..., SOFTMAX_MAX_MIN/MAX,
SOFTMAX_EXP_MIN/MAX, SOFTMAX_SUM_MIN/MAX), so the build script can switch
between static and profiled sources with no other changes.

Usage:
    eval "$(python3 load_profiled_ranges_n.py wider_mlp_profiled_ranges.json \
        --relu-tensor relu1_out \
        --relu-tensor relu2_out \
        --logits-tensor gemm3_out \
        --num-classes 10)"
"""

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("profiled_json")
    parser.add_argument("--relu-tensor", action="append", required=True,
                         metavar="TENSOR_NAME",
                         help="ONNX tensor name for a ReLU layer's output, in order "
                              "(layer 1 first). Repeatable.")
    parser.add_argument("--logits-tensor", required=True,
                         metavar="TENSOR_NAME",
                         help="ONNX tensor name for the final layer's logits "
                              "(Softmax's max-reduction input, no ReLU applied).")
    parser.add_argument("--num-classes", type=int, required=True,
                         help="Number of output classes, for the Softmax "
                              "sum-of-exponentials structural bound [1, N].")
    args = parser.parse_args()

    with open(args.profiled_json) as f:
        ranges = json.load(f)

    def get(name):
        if name not in ranges:
            print(f"ERROR: tensor '{name}' not found in {args.profiled_json}. "
                  f"Available tensors: {list(ranges.keys())}", file=sys.stderr)
            sys.exit(1)
        return ranges[name]["min"], ranges[name]["max"]

    print(f"# Loaded PROFILED (empirical) ranges from {args.profiled_json}",
          file=sys.stderr)

    exports = []
    for i, tensor_name in enumerate(args.relu_tensor, start=1):
        vmin, vmax = get(tensor_name)
        print(f"# ReLU{i} ({tensor_name}): [{vmin}, {vmax}]", file=sys.stderr)
        exports.append((f"RELU{i}_MIN", vmin))
        exports.append((f"RELU{i}_MAX", vmax))

    logits_min, logits_max = get(args.logits_tensor)
    print(f"# logits ({args.logits_tensor}): [{logits_min}, {logits_max}]",
          file=sys.stderr)
    exports.append(("SOFTMAX_MAX_MIN", logits_min))
    exports.append(("SOFTMAX_MAX_MAX", logits_max))

    # Structural, not input-dependent -- same reasoning as
    # compute_layer_ranges_n.py: one softmax term is always exactly
    # exp(0)=1 (the max element itself), others in [0,1].
    exports.append(("SOFTMAX_EXP_MIN", 0.0))
    exports.append(("SOFTMAX_EXP_MAX", 1.0))
    exports.append(("SOFTMAX_SUM_MIN", 1.0))
    exports.append(("SOFTMAX_SUM_MAX", float(args.num_classes)))

    for name, value in exports:
        print(f"export {name}={value!r}")


if __name__ == "__main__":
    main()
