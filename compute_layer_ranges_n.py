#!/usr/bin/env python3
"""
Compute derived intermediate-layer ranges (post-ReLU activation for each
hidden layer, final logits for Softmax) via interval arithmetic, for an
ARBITRARY chain of dense (Gemm + ReLU, except the last layer which has no
activation) layers. Generalizes the original two-layer (MNIST fc1/fc2)
version to any number of layers.

Prints bash `export VAR=value` statements to stdout, meant to be consumed via:
    eval "$(python3 compute_layer_ranges_n.py wider_mlp_ranges.json \
        --input-name image \
        --layer fc1.weight:fc1.bias:784 \
        --layer fc2.weight:fc2.bias:1024 \
        --layer fc3.weight:fc3.bias:512 \
        --num-classes 10)"

Emits (for a 3-layer chain as above):
    RELU1_MIN / RELU1_MAX   -- post-ReLU output of layer 1 (fc1)
    RELU2_MIN / RELU2_MAX   -- post-ReLU output of layer 2 (fc2)
    SOFTMAX_MAX_MIN / SOFTMAX_MAX_MAX  -- final layer's logits (fc3, no ReLU)
    SOFTMAX_EXP_MIN / SOFTMAX_EXP_MAX  -- always [0,1], structural
    SOFTMAX_SUM_MIN / SOFTMAX_SUM_MAX  -- always [1, num_classes], structural
(The last hidden layer's index is len(layers)-1; only layers 1..N-1 get a
RELU_i pair -- the final layer's output is the logits, not a ReLU output.)
"""

import argparse
import json
import sys


def interval_mult(a_min, a_max, b_min, b_max):
    corners = [a_min * b_min, a_min * b_max, a_max * b_min, a_max * b_max]
    return min(corners), max(corners)


def dense_layer_range(in_min, in_max, w_min, w_max, b_min, b_max, n_terms):
    term_min, term_max = interval_mult(in_min, in_max, w_min, w_max)
    sum_min, sum_max = n_terms * term_min, n_terms * term_max
    return sum_min + b_min, sum_max + b_max


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ranges_json")
    parser.add_argument("--input-name", default="image")
    parser.add_argument("--layer", action="append", required=True,
                         metavar="WEIGHT_NAME:BIAS_NAME:REDUCTION_N",
                         help="One dense layer, in order. Repeatable. "
                              "REDUCTION_N is that layer's input feature count "
                              "(the reduction dimension of its matmul). "
                              "Example: --layer fc1.weight:fc1.bias:784")
    parser.add_argument("--num-classes", type=int, required=True,
                         help="Number of output classes, for the Softmax "
                              "sum-of-exponentials structural bound [1, N].")
    args = parser.parse_args()

    with open(args.ranges_json) as f:
        ranges = json.load(f)

    def get(name):
        if name not in ranges:
            print(f"ERROR: '{name}' not found in {args.ranges_json}", file=sys.stderr)
            sys.exit(1)
        return ranges[name]["min"], ranges[name]["max"]

    layers = []
    for spec in args.layer:
        try:
            w_name, b_name, n_str = spec.split(":")
            layers.append((w_name, b_name, int(n_str)))
        except ValueError:
            print(f"ERROR: could not parse --layer spec {spec!r}, "
                  f"expected WEIGHT_NAME:BIAS_NAME:REDUCTION_N", file=sys.stderr)
            sys.exit(1)

    cur_min, cur_max = get(args.input_name)
    print(f"# input range: [{cur_min}, {cur_max}]", file=sys.stderr)

    exports = []
    for i, (w_name, b_name, n) in enumerate(layers, start=1):
        w_min, w_max = get(w_name)
        b_min, b_max = get(b_name)
        pre_min, pre_max = dense_layer_range(cur_min, cur_max, w_min, w_max,
                                             b_min, b_max, n)
        is_last = (i == len(layers))
        if is_last:
            print(f"# layer {i} ({w_name}) logits range: [{pre_min}, {pre_max}]",
                  file=sys.stderr)
            exports.append(("SOFTMAX_MAX_MIN", pre_min))
            exports.append(("SOFTMAX_MAX_MAX", pre_max))
        else:
            relu_min, relu_max = max(pre_min, 0.0), max(pre_max, 0.0)
            print(f"# layer {i} ({w_name}) pre-activation: [{pre_min}, {pre_max}]",
                  file=sys.stderr)
            print(f"# layer {i} ({w_name}) post-ReLU:      [{relu_min}, {relu_max}]",
                  file=sys.stderr)
            exports.append((f"RELU{i}_MIN", relu_min))
            exports.append((f"RELU{i}_MAX", relu_max))
            cur_min, cur_max = relu_min, relu_max

    exports.append(("SOFTMAX_EXP_MIN", 0.0))
    exports.append(("SOFTMAX_EXP_MAX", 1.0))
    exports.append(("SOFTMAX_SUM_MIN", 1.0))
    exports.append(("SOFTMAX_SUM_MAX", float(args.num_classes)))

    for name, value in exports:
        print(f"export {name}={value!r}")


if __name__ == "__main__":
    main()
