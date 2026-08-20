#!/usr/bin/env python3
"""
Compute derived intermediate-layer ranges (post-ReLU activation, fc2 logits)
via interval arithmetic over the model's known input/weight/bias ranges
(from extract_ranges.py's JSON output).

Prints bash `export VAR=value` statements to stdout, meant to be consumed via:
    eval "$(python3 compute_layer_ranges.py mnist_ranges.json \
        --fc1-weight fc1.weight --fc1-bias fc1.bias --fc1-n 196 \
        --fc2-weight fc2.weight --fc2-bias fc2.bias --fc2-n 128 \
        --input-name image)"
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ranges_json")
    parser.add_argument("--input-name", default="image")
    parser.add_argument("--fc1-weight", default="fc1.weight")
    parser.add_argument("--fc1-bias", default="fc1.bias")
    parser.add_argument("--fc1-n", type=int, required=True,
                         help="fc1's reduction dimension (number of input features)")
    parser.add_argument("--fc2-weight", default="fc2.weight")
    parser.add_argument("--fc2-bias", default="fc2.bias")
    parser.add_argument("--fc2-n", type=int, required=True,
                         help="fc2's reduction dimension (fc1's output size)")
    args = parser.parse_args()

    with open(args.ranges_json) as f:
        ranges = json.load(f)

    def get(name):
        if name not in ranges:
            print(f"ERROR: '{name}' not found in {args.ranges_json}", file=sys.stderr)
            sys.exit(1)
        return ranges[name]["min"], ranges[name]["max"]

    in_min, in_max = get(args.input_name)
    w1_min, w1_max = get(args.fc1_weight)
    b1_min, b1_max = get(args.fc1_bias)

    fc1_min, fc1_max = dense_layer_range(in_min, in_max, w1_min, w1_max,
                                          b1_min, b1_max, args.fc1_n)
    relu_min, relu_max = max(fc1_min, 0.0), max(fc1_max, 0.0)

    w2_min, w2_max = get(args.fc2_weight)
    b2_min, b2_max = get(args.fc2_bias)
    fc2_min, fc2_max = dense_layer_range(relu_min, relu_max, w2_min, w2_max,
                                          b2_min, b2_max, args.fc2_n)

    print(f"# input range:  [{in_min}, {in_max}]", file=sys.stderr)
    print(f"# fc1 pre-activation range: [{fc1_min}, {fc1_max}]", file=sys.stderr)
    print(f"# fc1 post-ReLU range:      [{relu_min}, {relu_max}]", file=sys.stderr)
    print(f"# fc2 (logits) range:       [{fc2_min}, {fc2_max}]", file=sys.stderr)

    print(f"export MAXPOOL_MIN={in_min!r}")
    print(f"export MAXPOOL_MAX={in_max!r}")
    print(f"export RELU_MIN={relu_min!r}")
    print(f"export RELU_MAX={relu_max!r}")
    print(f"export SOFTMAX_MAX_MIN={fc2_min!r}")
    print(f"export SOFTMAX_MAX_MAX={fc2_max!r}")
    print(f"export SOFTMAX_EXP_MIN=0.0")
    print(f"export SOFTMAX_EXP_MAX=1.0")


if __name__ == "__main__":
    main()
