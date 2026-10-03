#!/usr/bin/env python3
"""
Generate resnet20.cpp: a C++ driver for the resnet20_cifar10 model, using
a real CIFAR-10 test image (plain [0,1] pixel scaling, channels-first
[1,3,32,32] input, matching train_resnet20_cifar10.py's own preprocessing).

Usage:
    python3 generate_resnet20_driver.py --out resnet20.cpp
"""

import argparse
import sys

import numpy as np


DRIVER_TEMPLATE = r"""#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cstdint>

#include "OnnxMlirRuntime.h"

// Declare the inference entry point.
extern "C" OMTensorList *run_main_graph(OMTensorList *);

// Real CIFAR-10 test image, channels-first (3, 32, 32), scaled to [0,1]
// (plain pixel/255.0 -- matches train_resnet20_cifar10.py's
// transforms.ToTensor(), no further normalization).
static float img_data[] = {
@@IMG_DATA@@
};

// CIFAR-10 class names, in the standard order.
static const char *CLASS_NAMES[] = {
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
};

// Returns the current monotonic time in milliseconds.
static double now_ms() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

int main(int argc, char **argv) {
  // Number of timed inference iterations. Override with:
  //   ./resnet20_riscv_rvv_taffo <iterations>
  int iterations = 100;
  if (argc > 1) {
    iterations = atoi(argv[1]);
    if (iterations <= 0) iterations = 1;
  }

  int64_t rank = 4;
  int64_t shape[] = {1, 3, 32, 32};

  double *times_ms = new double[iterations];

  for (int iter = 0; iter < iterations; iter++) {
    int inputNum = 1;
    OMTensor *inputTensors[inputNum];
    OMTensor *tensor = omTensorCreate(img_data, shape, rank, ONNX_TYPE_FLOAT);
    inputTensors[0] = tensor;
    OMTensorList *tensorListIn = omTensorListCreate(inputTensors, inputNum);

    double t0 = now_ms();
    OMTensorList *tensorListOut = run_main_graph(tensorListIn);
    double t1 = now_ms();
    times_ms[iter] = t1 - t0;

    omTensorListDestroy(tensorListIn);

    // The model defines one output of type tensor<1x10xf32>.
    OMTensor *y = omTensorListGetOmtByIndex(tensorListOut, 0);
    float *prediction = (float *)omTensorGetDataPtr(y);

    if (iter == 0) {
      int digit = -1;
      float prob = 0.f;
      for (int i = 0; i < 10; i++) {
        printf("prediction[%d] = %f (%s)\n", i, prediction[i], CLASS_NAMES[i]);
        if (prediction[i] > prob) {
          digit = i;
          prob = prediction[i];
        }
      }
      printf("The predicted class is %d (%s)\n", digit,
             digit >= 0 ? CLASS_NAMES[digit] : "?");
    }

    omTensorListDestroy(tensorListOut);
  }

  double sum = 0.0;
  double min_ms = times_ms[0];
  double max_ms = times_ms[0];
  for (int i = 0; i < iterations; i++) {
    sum += times_ms[i];
    if (times_ms[i] < min_ms) min_ms = times_ms[i];
    if (times_ms[i] > max_ms) max_ms = times_ms[i];
  }
  double mean_ms = sum / iterations;

  printf("\n--- Inference timing (%d iteration%s) ---\n", iterations,
         iterations == 1 ? "" : "s");
  printf("min:  %.4f ms\n", min_ms);
  printf("mean: %.4f ms\n", mean_ms);
  printf("max:  %.4f ms\n", max_ms);

  delete[] times_ms;
  return 0;
}
"""


def fetch_real_image(seed=0, index=None):
    from torchvision import datasets
    print("Fetching CIFAR-10 test data (cached after first run)...", file=sys.stderr)
    test_set = datasets.CIFAR10(root="./cifar10_data", train=False, download=True)

    if index is None:
        rng = np.random.default_rng(seed)
        index = int(rng.integers(0, len(test_set)))

    # test_set.data is (N, 32, 32, 3) uint8, HWC.
    image_hwc = test_set.data[index].astype(np.float32)
    label = test_set.targets[index]
    class_names = test_set.classes
    print(f"Using test image index {index}, true label: {label} ({class_names[label]})",
          file=sys.stderr)

    image_chw = image_hwc.transpose(2, 0, 1) / 255.0  # HWC -> CHW, plain [0,1] scaling
    return image_chw.astype(np.float32)


def format_c_array(values, per_line=8):
    flat = values.flatten()
    lines = []
    for i in range(0, len(flat), per_line):
        chunk = flat[i:i + per_line]
        lines.append("    " + ", ".join(f"{v:.10f}f" for v in chunk) + ",")
    lines[-1] = lines[-1].rstrip(",")  # no trailing comma on the last value
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="resnet20.cpp")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--index", type=int, default=None,
                         help="Specific CIFAR-10 test image index to use (default: random)")
    args = parser.parse_args()

    image = fetch_real_image(seed=args.seed, index=args.index)
    assert image.shape == (3, 32, 32), f"unexpected image shape {image.shape}"

    img_data_str = format_c_array(image)
    driver_code = DRIVER_TEMPLATE.replace("@@IMG_DATA@@", img_data_str)

    with open(args.out, "w") as f:
        f.write(driver_code)

    print(f"Wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
