#!/bin/bash
#
# build_mnist_riscv.sh
# ------------------------------------------------------------
# Build script for compiling an ONNX MNIST model into a
# RISC-V shared library using ONNX-MLIR, LLVM, and the RISC-V GNU toolchain.
#
# This version enables:
#   - Double-precision floating point support
#   - PIC (Position Independent Code)
#
# Build pipeline:
#   1. Compile the ONNX model into a RISC-V object file
#      using ONNX-MLIR.
#   2. Compile the ONNX-MLIR runtime support sources.
#   3. Link everything into a shared library (.so).
#
# Result:
#   - mnist_riscv.so	(shared library)
#   - mnist_riscv	(test executable)
#
# Requirements:
#   - ONNX-MLIR installed and available in PATH
#   - LLVM tools installed:
#       * mlir-translate
#       * llc
#   - RISC-V GNU toolchain installed and available in PATH
#   - A valid RISC-V Linux target environment
#
# Example usage:
#   chmod +x build_mnist_riscv.sh
#   ./build_mnist_riscv.sh
# ------------------------------------------------------------

# Exit immediately if any command fails.
# This prevents continuing with partial or invalid builds.
set -e

# ============================================================
# Configuration Section
# ============================================================

# Path to the ONNX model to compile.
# This example uses the MNIST model included in ONNX-MLIR docs.
ONNX_MODEL=~/onnx-mlir/onnx-mlir/docs/mnist_example/mnist.onnx

# Path to ONNX-MLIR public header files.
# Required when compiling runtime support code.
ONNX_MLIR_INCLUDE=~/onnx-mlir/onnx-mlir/include

# Path to ONNX-MLIR runtime source files.
# These implement tensor management and runtime helpers.
RUNTIME_DIR=~/onnx-mlir/onnx-mlir/riscv_build/src/Runtime

# Path to ONNX-MLIR support library source files.
# These provide additional low-level utility functions.
SUPPORT_DIR=~/onnx-mlir/onnx-mlir/riscv_build/src/Support

# RISC-V GCC cross-compiler executable.
# Used for compiling and linking generated code.
RISCV_GCC=~/riscv-gnu-toolchain/riscv/bin/riscv64-unknown-linux-gnu-gcc

# Name of the final shared library output.
OUTPUT=mnist_riscv.so

# ============================================================
# Cleanup Section
# ============================================================

# Remove previously generated object files and shared libraries.
# Uncomment if you want a clean rebuild every time.
#
# rm -f *.o *.so

# ============================================================
# Step 1 - Compile ONNX model with ONNX-MLIR
# ============================================================

echo "- Compiling ONNX model:"

# Compile the ONNX model into a RISC-V object file.
#
# Important options:
#   --mtriple=riscv64-unknown-linux-gnu
#       Target architecture triple.
#
#   -O3
#       Enable high-level compiler optimizations.
#
#   --EmitObj
#       Generate an object file (.o) instead of executable code.
#
#   -o mnist_riscv
#       Output filename prefix.
#
#   -Xllc "-mattr=+d"
#       Enable the RISC-V double-precision floating-point extension.
#
onnx-mlir $ONNX_MODEL \
  --mtriple=riscv64-unknown-linux-gnu \
  -O3 \
  --EmitObj \
  -o mnist_riscv \
  -Xllc "-mattr=+d"

# Output generated:
#   mnist_riscv.o

# ============================================================
# Step 2 - Compile ONNX-MLIR runtime sources
# ============================================================

echo "- Compiling runtime (C):"

# Compile runtime and support C source files into object files.
#
# Runtime files are required because the generated ONNX model
# depends on tensor/runtime helper functions provided by
# ONNX-MLIR.
#
# Compiler options:
#   -c
#       Compile only (do not link yet).
#
#   -I
#       Add include directory for ONNX-MLIR headers.
#
#   -march=rv64gc
#       Target RISC-V ISA:
#         rv64  = 64-bit RISC-V
#         g     = general-purpose ISA extensions
#         c     = compressed instruction extension
#
#   -mabi=lp64d
#       ABI using:
#         lp64 = 64-bit long/pointer
#         d    = double-precision floating point ABI
#
#   -fPIC
#       Generate position-independent code required for
#       shared libraries.The future of building 
#
$RISCV_GCC -c \
  $RUNTIME_DIR/*.c \
  $SUPPORT_DIR/*.c \
  -I $ONNX_MLIR_INCLUDE \
  -march=rv64gc -mabi=lp64d \
  -fPIC

# Example generated object files:
#   OMTensor.o
#   OMTensorList.o

# ============================================================
# Step 3 - Link shared library
# ============================================================

echo "- Linking shared library:"

# Link all object files into a shared library.
#
# Options:
#   -shared
#       Produce a shared object (.so).
#
#   *.o
#       Include all object files generated previously.
#
#   -o $OUTPUT
#       Name of the final shared library.
#
$RISCV_GCC -shared \
  *.o \
  -o $OUTPUT \
  -march=rv64gc -mabi=lp64d

# ============================================================
# Build Complete
# ============================================================

echo "- Build complete: $OUTPUT"
