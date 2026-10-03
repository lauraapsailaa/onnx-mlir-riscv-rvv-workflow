#!/usr/bin/env python3
"""
Profile a model's TRUE intermediate tensor ranges by exposing every
intermediate output in the ONNX graph and running real test data through
it with onnxruntime, recording observed min/max per tensor.

This replaces the hand-derived interval-arithmetic bounds from
compute_layer_ranges.py with EMPIRICAL ranges observed on real data --
the "profile-guided" approach, as opposed to the "manual/ideal
annotation" approach used earlier in this pipeline.

IMPORTANT -- preprocessing must match the actual deployed input exactly.
By default this script applies standard MNIST normalization
((pixel/255 - mean) / std, mean=0.1307, std=0.3081), matching the
hardcoded test image already used in mnist.cpp (values roughly in
[-0.4242, 2.82], NOT [0,1]). If your deployment uses different
preprocessing, adjust --input-mean/--input-std accordingly, or pass
--no-normalize to just use raw [0,1] pixel values.

Usage:
    python3 profile_ranges.py mnist.onnx \
        --input-name image \
        --num-images 200 \
        --json-out mnist_profiled_ranges.json
"""

import argparse
import json
import sys

import numpy as np
import onnx
import onnxruntime as ort


def load_mnist_test_images(num_images, seed=0):
    """
    Load real MNIST test images via sklearn's fetch_openml (downloads and
    caches the dataset on first use). Returns raw pixel values in [0, 255],
    shape (num_images, 28, 28).
    """
    from sklearn.datasets import fetch_openml
    print(f"Fetching MNIST test data (this may take a while on first run, "
          f"cached afterward)...", file=sys.stderr)
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True,
                        as_frame=False, parser="auto")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(num_images, len(X)), replace=False)
    images = X[idx].reshape(-1, 28, 28).astype(np.float32)
    return images


def load_cifar10_test_images(num_images, seed=0):
    """
    Load real CIFAR-10 test images via torchvision (downloads and caches
    on first use), matching train_resnet20_cifar10.py's own data source.
    Returns raw pixel values in [0, 255], shape (num_images, 3, 32, 32)
    (channels-first, matching the model's actual input layout).
    """
    from torchvision import datasets
    print("Fetching CIFAR-10 test data (cached after first run)...",
          file=sys.stderr)
    test_set = datasets.CIFAR10(root="./cifar10_data", train=False,
                                download=True)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(test_set), size=min(num_images, len(test_set)),
                     replace=False)
    # test_set.data is (N, 32, 32, 3) uint8, HWC -- transpose to (N, 3, 32, 32)
    images = test_set.data[idx].astype(np.float32)
    images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
    return images


def preprocess_mnist(images, mean, std, normalize=True, flatten=False):
    """images: (N, 28, 28) raw pixel values in [0,255]."""
    x = images / 255.0
    if normalize:
        x = (x - mean) / std
    x = x.astype(np.float32)
    if flatten:
        return x.reshape(-1, 1, 784)
    return x.reshape(-1, 1, 1, 28, 28)


def preprocess_cifar10(images, mean, std, normalize=True):
    """images: (N, 3, 32, 32) raw pixel values in [0,255], channels-first."""
    x = images / 255.0
    if normalize:
        # mean/std would need to be per-channel for a real CIFAR normalization
        # scheme; --no-normalize (plain [0,1]) is what train_resnet20_cifar10.py
        # actually uses, so that's the expected default for this dataset.
        x = (x - mean) / std
    x = x.astype(np.float32)
    return x.reshape(-1, 1, 3, 32, 32)


def expose_all_intermediate_outputs(model_path, exposed_model_path):
    """
    Load an ONNX model and add every intermediate tensor as an additional
    graph output, so onnxruntime will return them alongside the normal
    final output(s). Returns the list of exposed tensor names.
    """
    model = onnx.load(model_path)
    model = onnx.shape_inference.infer_shapes(model)

    # Look up each tensor's REAL inferred dtype (shape inference computes
    # this) instead of assuming FLOAT for everything -- some intermediate
    # tensors (e.g. Shape/ConstantOfShape/index computations) are
    # genuinely int64 or another type, and onnxruntime will refuse to
    # load a model where a declared output type doesn't match reality.
    dtype_by_name = {}
    for vi in model.graph.value_info:
        dtype_by_name[vi.name] = vi.type.tensor_type.elem_type
    for vi in model.graph.input:
        dtype_by_name[vi.name] = vi.type.tensor_type.elem_type
    for init in model.graph.initializer:
        dtype_by_name[init.name] = init.data_type

    existing_output_names = {o.name for o in model.graph.output}
    exposed_names = []

    # Every node's output not already a graph output is a candidate.
    for node in model.graph.node:
        for out_name in node.output:
            if out_name and out_name not in existing_output_names:
                exposed_names.append(out_name)
                existing_output_names.add(out_name)
                elem_type = dtype_by_name.get(out_name, onnx.TensorProto.FLOAT)
                value_info = onnx.helper.make_tensor_value_info(
                    out_name, elem_type, None)
                model.graph.output.append(value_info)

    onnx.save(model, exposed_model_path)
    print(f"Exposed {len(exposed_names)} intermediate tensor(s) as outputs",
          file=sys.stderr)
    return exposed_names


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_path")
    parser.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist",
                         help="Which real test dataset to profile over (default: mnist)")
    parser.add_argument("--input-name", default="image")
    parser.add_argument("--num-images", type=int, default=200,
                         help="Number of real test images to profile over (default: 200)")
    parser.add_argument("--input-mean", type=float, default=0.1307,
                         help="Normalization mean (default: 0.1307, standard MNIST)")
    parser.add_argument("--input-std", type=float, default=0.3081,
                         help="Normalization std (default: 0.3081, standard MNIST)")
    parser.add_argument("--no-normalize", action="store_true",
                         help="Skip mean/std normalization, use raw [0,1] pixel values instead")
    parser.add_argument("--flatten", action="store_true",
                         help="Reshape each image to a flat (1, 784) input instead of the "
                              "default (1, 1, 28, 28) -- use for models expecting a flat "
                              "input, e.g. wider_mlp.onnx (no MaxPool/Conv, straight to Gemm)")
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    exposed_model_path = "/tmp/_profiling_exposed_model.onnx"
    exposed_names = expose_all_intermediate_outputs(args.model_path, exposed_model_path)

    print(f"Loading real {args.dataset.upper()} test images...", file=sys.stderr)
    if args.dataset == "cifar10":
        images = load_cifar10_test_images(args.num_images, seed=args.seed)
        batch = preprocess_cifar10(images, args.input_mean, args.input_std,
                                   normalize=not args.no_normalize)
    else:
        images = load_mnist_test_images(args.num_images, seed=args.seed)
        batch = preprocess_mnist(images, args.input_mean, args.input_std,
                                 normalize=not args.no_normalize, flatten=args.flatten)
    print(f"Profiling over {batch.shape[0]} image(s), "
          f"input range in this batch: [{batch.min():.4f}, {batch.max():.4f}]",
          file=sys.stderr)

    session = ort.InferenceSession(exposed_model_path)
    output_names = [o.name for o in session.get_outputs()]

    # Running images one at a time keeps peak memory low and matches how
    # the real deployed model is actually invoked (single-image inference).
    running_min = {}
    running_max = {}

    for i in range(batch.shape[0]):
        single_input = batch[i]  # shape (1, 1, 28, 28)
        outputs = session.run(output_names, {args.input_name: single_input})
        for name, value in zip(output_names, outputs):
            arr = np.asarray(value, dtype=np.float64)
            vmin, vmax = float(arr.min()), float(arr.max())
            if name not in running_min:
                running_min[name] = vmin
                running_max[name] = vmax
            else:
                running_min[name] = min(running_min[name], vmin)
                running_max[name] = max(running_max[name], vmax)

    result = {
        name: {"min": running_min[name], "max": running_max[name]}
        for name in output_names
    }
    # The graph's own input isn't a node output, so it's never captured by
    # the loop above -- add its observed range explicitly.
    result[args.input_name] = {
        "min": float(batch.min()),
        "max": float(batch.max()),
    }

    with open(args.json_out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nWrote {args.json_out} with ranges for {len(result)} tensor(s) "
          f"(profiled over {batch.shape[0]} real images)", file=sys.stderr)
    print(f"Input '{args.input_name}' observed range: "
          f"[{result.get(args.input_name, {}).get('min', 'N/A')}, "
          f"{result.get(args.input_name, {}).get('max', 'N/A')}]", file=sys.stderr)


if __name__ == "__main__":
    main()