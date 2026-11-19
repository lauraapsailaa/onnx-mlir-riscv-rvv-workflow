#!/bin/bash
#
# build_resnet20_rvv_fast.sh
# ------------------------------------------------------------
# Build script for compiling an ONNX RESNET20 model into a
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
#   2. Translate MLIR -> LLVM IR
#   3. Compile LLVM IR -> RISC-V assembly using llc
#   4. Assemble into object code
#   5. Compile ONNX-MLIR runtime support
#   6. Link all objects into a shared library
#   7. Build a standalone inference executable (statically linked)
#
# All generated files are written to a dedicated BUILD_DIR, kept separate
# from the plain and TAFFO build scripts' own output directories.
#
# Final outputs (inside BUILD_DIR):
#   - resnet20_riscv_rvv_float_fast.so      (shared library)
#   - resnet20_riscv_rvv_float_fast          (test executable)
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
#   chmod +x build_riscv_rvv_resnet20.sh
#   ./build_riscv_rvv_resnet20.sh
# ------------------------------------------------------------

# Exit immediately if any command fails.
# Prevents invalid partial builds.
set -e

# ============================================================
# Configuration Section
# ============================================================

RESNET20_DIR=~/onnx-mlir/onnx-mlir/docs/other_examples/ResNet20-noBN-cifar10
BUILD_DIR=$RESNET20_DIR/resnet20_rvv_fast

mkdir -p $BUILD_DIR

# Path to the ONNX RESNET20 model.
ONNX_MODEL=$RESNET20_DIR/resnet20_cifar10.onnx

# Path to ONNX-MLIR public headers.
ONNX_MLIR_INCLUDE=~/onnx-mlir/onnx-mlir/include

# Path to ONNX-MLIR runtime source directory.
RUNTIME_DIR=~/onnx-mlir/onnx-mlir/riscv_build/src/Runtime

# Path to ONNX-MLIR support source directory.
SUPPORT_DIR=~/onnx-mlir/onnx-mlir/riscv_build/src/Support

# RISC-V cross compiler executable.
RISCV_GCC=~/riscv-gnu-toolchain/riscv/bin/riscv64-unknown-linux-gnu-gcc

# Name of the generated shared library.
OUTPUT=resnet20_riscv_rvv_float_fast.so

# Directory containing the driver source (resnet20.cpp) -- same directory as
# this script, independent of BUILD_DIR.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVER_SRC=$SCRIPT_DIR/resnet20.cpp

echo "BUILD_DIR:  $BUILD_DIR"
echo "DRIVER_SRC: $DRIVER_SRC"
echo ""

# Every subsequent command that writes output with a relative path will
# now write into BUILD_DIR, not wherever the script happened to be invoked
# from -- keeping this build fully self-contained.
cd $BUILD_DIR

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
  -o resnet20_riscv_rvv_float_fast \
  -Xllc "-march=riscv64" \
  -Xllc "-mcpu=generic-rv64" \
  -Xllc "-mattr=+d,+v"

# Generated intermediate files may include:
#   resnet20_riscv_rvv_float_fast.onnx.mlir

# ============================================================
# Step 2 - Translate MLIR to LLVM IR
# ============================================================

# Convert ONNX-MLIR generated MLIR into standard LLVM IR.
#
# Output:
#   resnet20_riscv_rvv_float_fast.ll
#
mlir-translate \
  --mlir-to-llvmir \
  resnet20_riscv_rvv_float_fast.onnx.mlir \
  -o resnet20_riscv_rvv_float_fast.ll

# ============================================================
# Step 2b - Run LLVM's real optimizer (enables the Loop Vectorizer)
# ============================================================

# llc alone does NOT run LLVM's middle-end optimization passes, including
# the Loop Vectorizer (confirmed empirically: zero vectorize remarks from
# llc even with -O3). The vectorizer only runs as part of opt's own
# pipeline -- run it explicitly here for a fair, consistent comparison
# against the TAFFO build (which also now runs this same step).
# Allow float reassociation: mark every fadd/fsub/fmul with reassoc/nsz/contract so the
# Loop Vectorizer is allowed to vectorize float reductions (a fairer float baseline;
# without it strict FP ordering keeps the reduction loops scalar).
sed -E -i 's/\b(fadd|fsub|fmul) (float|double)/\1 reassoc nsz contract \2/' resnet20_riscv_rvv_float_fast.ll
echo "fast-math flags added to $(grep -c 'reassoc' resnet20_riscv_rvv_float_fast.ll) instruction(s)"

opt -O3 -mtriple=riscv64 -mattr=+v,+d,+m \
  -S resnet20_riscv_rvv_float_fast.ll \
  -o resnet20_riscv_rvv_float_fast_opt.ll

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
#   -mattr=+v,+d,+m
#       Enable vector, double-precision, and multiply extensions.
#
#   -relocation-model=pic
#       Generate position-independent assembly suitable for
#       shared libraries.
#
# Output:
#   resnet20_riscv_rvv_float_fast.s
#
llc \
  -march=riscv64 \
  -mattr=+v,+d,+m \
  -relocation-model=pic \
  resnet20_riscv_rvv_float_fast_opt.ll \
  -o resnet20_riscv_rvv_float_fast.s

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
  resnet20_riscv_rvv_float_fast.s \
  -march=rv64gcv \
  -mabi=lp64d \
  -fPIC \
  -o resnet20_riscv_rvv_float_fast.o

# Generated:
#   resnet20_riscv_rvv_float_fast.o

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
riscv64-unknown-linux-gnu-gcc -c \
  $RUNTIME_DIR/*.c \
  $SUPPORT_DIR/*.c \
  -I $ONNX_MLIR_INCLUDE \
  -march=rv64gcv \
  -mabi=lp64d \
  -fPIC

mkdir -p runtime_objs
find . -maxdepth 1 -name '*.o' ! -name 'resnet20_riscv_rvv_float_fast.o' -exec mv {} runtime_objs/ \;

# Example generated objects:
#   OMTensor.o
#   OMTensorList.o

# ============================================================
# Step 6 - Link Shared Library
# ============================================================

echo "- Linking shared library: "

# Link this run's model object plus this run's runtime objects only --
# never a blind "*.o" glob, which could also pick up unrelated .o files
# from the plain/TAFFO builds if they ever shared a directory (these
# export the same symbol names, causing "multiple definition" link errors).
riscv64-unknown-linux-gnu-gcc -shared \
  resnet20_riscv_rvv_float_fast.o \
  runtime_objs/*.o \
  -o $OUTPUT \
  -march=rv64gc \
  -mabi=lp64d

echo "- Build complete: $OUTPUT"

# ============================================================
# Step 7 - Build Standalone Test Executable
# ============================================================

# Link STATICALLY, directly against the object files -- NOT the .so. A
# dynamically-linked executable bakes in the .so's absolute build-time
# path, which the dynamic linker then looks for verbatim on the target
# board (where it won't exist), causing "cannot open shared object file"
# at runtime. A single self-contained static binary avoids this entirely.
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
  $DRIVER_SRC \
  resnet20_riscv_rvv_float_fast.o \
  runtime_objs/*.o \
  -o resnet20_riscv_rvv_float_fast \
  -I $ONNX_MLIR_INCLUDE \
  -march=rv64gcv \
  -mabi=lp64d \
  -static

echo "- Build complete: $BUILD_DIR/resnet20_riscv_rvv_float_fast"
