#!/usr/bin/env python3
"""
Extract exact min/max ranges for every initializer (weight/bias tensor) in an
ONNX model. These ranges can be used to build TAFFO `taffo.cast2real`
annotations with "ideal" (ground-truth) bounds, per the manual-annotation
approach.

Usage:
    python3 extract_ranges.py path/to/model.onnx [--input-name X --input-min 0.0 --input-max 1.0]
"""

import argparse
import json
import sys

import numpy as np
import onnx
from onnx import numpy_helper


def extract_initializer_ranges(model_path):
    model = onnx.load(model_path)
    graph = model.graph

    results = {}
    for init in graph.initializer:
        arr = numpy_helper.to_array(init)
        arr = arr.astype(np.float64)  # avoid overflow/precision issues while reducing
        vmin = float(arr.min())
        vmax = float(arr.max())
        results[init.name] = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "min": vmin,
            "max": vmax,
            "num_elements": int(arr.size),
        }
    return results, graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", help="Path to the .onnx model file")
    parser.add_argument("--input-name", default=None,
                         help="Name of the graph's input tensor (default: first graph input)")
    parser.add_argument("--input-min", type=float, default=0.0,
                         help="Assumed minimum value for the model input (default: 0.0)")
    parser.add_argument("--input-max", type=float, default=1.0,
                         help="Assumed maximum value for the model input (default: 1.0)")
    parser.add_argument("--precision", type=float, default=0.01,
                         help="Default TAFFO precision/error term to use for all annotations (default: 0.01)")
    parser.add_argument("--json-out", default=None,
                         help="Optional path to write the full results as JSON")
    args = parser.parse_args()

    results, graph = extract_initializer_ranges(args.model_path)

    if not results:
        print(f"WARNING: no initializers (weights/biases) found in {args.model_path}", file=sys.stderr)

    # Determine the input tensor name
    graph_input_names = [inp.name for inp in graph.input]
    input_name = args.input_name or (graph_input_names[0] if graph_input_names else None)

    print("=" * 70)
    print(f"Model: {args.model_path}")
    print("=" * 70)
    print()
    print(f"Graph inputs: {graph_input_names}")
    print(f"Using input '{input_name}' with assumed range "
          f"[{args.input_min}, {args.input_max}] (precision={args.precision})")
    print()
    print(f"{'Tensor name':40s} {'shape':>18s} {'min':>14s} {'max':>14s}")
    print("-" * 90)
    for name, info in results.items():
        print(f"{name:40s} {str(info['shape']):>18s} "
              f"{info['min']:14.6f} {info['max']:14.6f}")

    print()
    print("=" * 70)
    print("Suggested taffo.cast2real bounds (copy/paste-ready dict)")
    print("=" * 70)
    # Determine the input tensor's shape from the graph's value_info, if available.
    input_shape = None
    for inp in graph.input:
        if inp.name == input_name:
            input_shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            break

    annotations = {}
    if input_name:
        annotations[input_name] = {
            "min": args.input_min,
            "max": args.input_max,
            "precision": args.precision,
            "shape": input_shape,
            "role": "input",
        }
    for name, info in results.items():
        annotations[name] = {
            "min": info["min"],
            "max": info["max"],
            "precision": args.precision,
            "shape": info["shape"],
            "role": "weight",
        }

    print(json.dumps(annotations, indent=2))

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(annotations, f, indent=2)
        print()
        print(f"Written to {args.json_out}")


if __name__ == "__main__":
    main()
