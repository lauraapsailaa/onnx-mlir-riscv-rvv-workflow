#!/bin/bash
#
# build_mnist_riscv.sh
# ------------------------------------------------------------
# Build script for compiling an ONNX MNIST model into a
# RISC-V Vector Extension (RVV) shared library using
# ONNX-MLIR, LLVM, and the RISC-V GNU toolchain.
#
# This version enables:
#   - RISC-V Vector Extension (RVV)
#   - Double-precision floating point support
#   - PIC (Position Independent Code)
#
# Build pipeline:
#   1. Generate MLIR/LLVM IR from ONNX using ONNX-MLIR
#   2. Translate MLIR → LLVM IR
#   3. Compile LLVM IR → RISC-V assembly using llc
#   4. Assemble into object code
#   5. Compile ONNX-MLIR runtime support
#   6. Link all objects into a shared library
#   7. Build a standalone inference executable
#
# Final outputs:
#   - mnist_riscv_rvv.so      (shared library)
#   - mnist_riscv_rvv         (test executable)
#
# Requirements:
#   - ONNX-MLIR installed
#   - LLVM tools installed:
#       * mlir-translate
#       * llc
#   - RISC-V GNU toolchain installed and available in PATH
#   - RVV-capable RISC-V target
#
# Example usage:
#   chmod +x build_mnist_riscv.sh
#   ./build_mnist_riscv.sh
# ------------------------------------------------------------

# Exit immediately if any command fails.
# Prevents invalid partial builds.
set -e

# ============================================================
# Configuration Section
# ============================================================

# Path to the ONNX MNIST model.
ONNX_MODEL=~/onnx-mlir/onnx-mlir/docs/mnist_example/mnist.onnx

# Path to ONNX-MLIR public headers.
ONNX_MLIR_INCLUDE=~/onnx-mlir/onnx-mlir/include

# Path to ONNX-MLIR runtime source directory.
RUNTIME_DIR=~/onnx-mlir/onnx-mlir/riscv_build/src/Runtime

# Path to ONNX-MLIR support source directory.
SUPPORT_DIR=~/onnx-mlir/onnx-mlir/riscv_build/src/Support

# RISC-V cross compiler executable.
RISCV_GCC=~/riscv-gnu-toolchain/riscv/bin/riscv64-unknown-linux-gnu-gcc

# Name of the generated shared library.
OUTPUT=mnist_riscv_rvv.so

# ============================================================
# LLVM Backend Configuration
# ============================================================

# Export LLVM target options.
#
# "+v" enables the RISC-V Vector Extension (RVV).
# This affects LLVM code generation.
#
export LLVM_TARGET_OPTIONS="-mattr=+v"

# ============================================================
# Cleanup Section
# ============================================================

# Remove previously generated object files and libraries.
# Uncomment for clean rebuilds.
#
# rm -f *.o *.so

# ============================================================
# Step 1 - Generate LLVM IR from ONNX
# ============================================================

echo "- Compiling ONNX model: "

# Use ONNX-MLIR to convert the ONNX model into MLIR/LLVM IR.
#
# Important options:
#
#   --mtriple=riscv64-unknown-linux-gnu
#       Target platform triple.
#
#   --EmitLLVMIR
#       Emit LLVM-compatible IR instead of directly generating
#       machine object code.
#
#   -Xllc "-march=riscv64"
#       LLVM backend target architecture.
#
#   -Xllc "-mcpu=generic-rv64"
#       Generic RV64 CPU target.
#
#   -Xllc "-mattr=+d,+v"
#       Enable:
#         +d = double-precision floating point
#         +v = RISC-V Vector Extension
#
onnx-mlir $ONNX_MODEL \
  --mtriple=riscv64-unknown-linux-gnu \
  --EmitLLVMIR \
  -o mnist_riscv_rvv \
  -Xllc "-march=riscv64" \
  -Xllc "-mcpu=generic-rv64" \
  -Xllc "-mattr=+d,+v"

# Generated intermediate files may include:
#   mnist_riscv_rvv.onnx.mlir
#   mnist_riscv_rvv.ll

# ============================================================
# Step 2 - Translate MLIR to LLVM IR
# ============================================================

# Convert ONNX-MLIR generated MLIR into standard LLVM IR.
#
# Output:
#   mnist_riscv_rvv.ll
#
mlir-translate \
  --mlir-to-llvmir \
  mnist_riscv_rvv.onnx.mlir \
  -o mnist_riscv_rvv.ll

# ============================================================
# Step 3 - Compile LLVM IR to RISC-V Assembly
# ============================================================

# Use LLVM static compiler (llc) to generate RISC-V assembly.
#
# Options:
#
#   -march=riscv64
#       Target RISC-V 64-bit architecture.
#
#   -mattr=+v,+d
#       Enable vector and double-precision extensions.
#
#   -relocation-model=pic
#       Generate position-independent assembly suitable for
#       shared libraries.
#
# Output:
#   mnist_riscv_rvv.s
#
llc \
  -march=riscv64 \
  -mattr=+v,+d \
  -relocation-model=pic \
  mnist_riscv_rvv.ll \
  -o mnist_riscv_rvv.s

# ============================================================
# Step 4 - Assemble RISC-V Object File
# ============================================================

# Assemble generated RISC-V assembly into an object file.
#
# Compiler flags:
#
#   -march=rv64gcv
#       RISC-V ISA with:
#         g = general ISA extensions
#         c = compressed instructions
#         v = vector extension
#
#   -mabi=lp64d
#       64-bit ABI with double-precision floating point.
#
#   -fPIC
#       Generate position-independent code.
#
riscv64-unknown-linux-gnu-gcc -c \
  mnist_riscv_rvv.s \
  -march=rv64gcv \
  -mabi=lp64d \
  -fPIC \
  -o mnist_riscv_rvv.o

# Generated:
#   mnist_riscv_rvv.o

# ============================================================
# Step 5 - Compile ONNX-MLIR Runtime Sources
# ============================================================

echo "- Compiling runtime (C): "

# Compile ONNX-MLIR runtime and support libraries.
#
# These sources provide:
#   - Tensor data structures
#   - Runtime memory management
#   - Utility helper functions
#
$RISCV_GCC -c \
  $RUNTIME_DIR/*.c \
  $SUPPORT_DIR/*.c \
  -I $ONNX_MLIR_INCLUDE \
  -march=rv64gcv \
  -mabi=lp64d \
  -fPIC

# Example generated objects:
#   OMTensor.o
#   OMTensorList.o

# ============================================================
# Step 6 - Link Shared Library
# ============================================================

echo "- Linking shared library: "

# Link all generated object files into a shared library.
#
# Notes:
#   -shared
#       Produce a .so shared object.
#
#   *.o
#       Include all object files in current directory.
#
$RISCV_GCC -shared \
  *.o \
  -o $OUTPUT \
  -march=rv64gc \
  -mabi=lp64d

echo "- Build complete: $OUTPUT"

# ============================================================
# Step 7 - Build Standalone Test Executable (optional)
# ============================================================

# Compile the MNIST inference test program and link it
# against the generated shared library.
#
# This executable can be used to run inference directly
# on the target RISC-V platform.
#
# Compiler options:
#
#   --std=c++11
#       Use C++11 standard.
#
#   -O3
#       Enable aggressive optimizations.
#
#   -I
#       Include ONNX-MLIR header directories.
#
#   -march=rv64gcv
#       Enable RVV instructions in generated executable.
#
riscv64-unknown-linux-gnu-g++ \
  --std=c++11 -O3 \
  mnist.cpp ./mnist_riscv_rvv.so \
  -o mnist_riscv_rvv \
  -I $ONNX_MLIR_INCLUDE \
  -I $ONNX_MLIR_BUILD_INCLUDE \
  -march=rv64gcv \
  -mabi=lp64d \
  -fPIC

# Final generated executable:
#   mnist_riscv_rvv
