#!/bin/bash
#
# build_riscv_rvv_wider_mlp.sh
# ------------------------------------------------------------
# Build script for compiling an ONNX WIDER_MLP model into a
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
#   - wider_mlp_riscv_rvv.so      (shared library)
#   - wider_mlp_riscv_rvv          (test executable)
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
#   chmod +x build_riscv_rvv_wider_mlp.sh
#   ./build_riscv_rvv_wider_mlp.sh
# ------------------------------------------------------------

# Exit immediately if any command fails.
# Prevents invalid partial builds.
set -e

# ============================================================
# Configuration Section
# ============================================================

WIDER_MLP_DIR=~/onnx-mlir/onnx-mlir/docs/other_examples/wider_mlp
BUILD_DIR=$WIDER_MLP_DIR/wider_mlp_rvv

mkdir -p $BUILD_DIR

# Path to the ONNX WIDER_MLP model.
ONNX_MODEL=$WIDER_MLP_DIR/wider_mlp.onnx

# Path to ONNX-MLIR public headers.
ONNX_MLIR_INCLUDE=~/onnx-mlir/onnx-mlir/include

# Path to ONNX-MLIR runtime source directory.
RUNTIME_DIR=~/onnx-mlir/onnx-mlir/riscv_build/src/Runtime

# Path to ONNX-MLIR support source directory.
SUPPORT_DIR=~/onnx-mlir/onnx-mlir/riscv_build/src/Support

# RISC-V cross compiler executable.
RISCV_GCC=~/riscv-gnu-toolchain/riscv/bin/riscv64-unknown-linux-gnu-gcc

# Name of the generated shared library.
OUTPUT=wider_mlp_riscv_rvv.so

# Directory containing the driver source (wider_mlp.cpp) -- same directory as
# this script, independent of BUILD_DIR.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVER_SRC=$SCRIPT_DIR/wider_mlp.cpp

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
  -o wider_mlp_riscv_rvv \
  -Xllc "-march=riscv64" \
  -Xllc "-mcpu=generic-rv64" \
  -Xllc "-mattr=+d,+v"

# Generated intermediate files may include:
#   wider_mlp_riscv_rvv.onnx.mlir

# ============================================================
# Step 2 - Translate MLIR to LLVM IR
# ============================================================

# Convert ONNX-MLIR generated MLIR into standard LLVM IR.
#
# Output:
#   wider_mlp_riscv_rvv.ll
#
mlir-translate \
  --mlir-to-llvmir \
  wider_mlp_riscv_rvv.onnx.mlir \
  -o wider_mlp_riscv_rvv.ll

# ============================================================
# Step 2b - Run LLVM's real optimizer (enables the Loop Vectorizer)
# ============================================================

# llc alone does NOT run LLVM's middle-end optimization passes, including
# the Loop Vectorizer (confirmed empirically: zero vectorize remarks from
# llc even with -O3). The vectorizer only runs as part of opt's own
# pipeline -- run it explicitly here for a fair, consistent comparison
# against the TAFFO build (which also now runs this same step).
opt -O3 -mtriple=riscv64 -mattr=+v,+d,+m \
  -S wider_mlp_riscv_rvv.ll \
  -o wider_mlp_riscv_rvv_opt.ll

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
#   wider_mlp_riscv_rvv.s
#
llc \
  -march=riscv64 \
  -mattr=+v,+d,+m \
  -relocation-model=pic \
  wider_mlp_riscv_rvv_opt.ll \
  -o wider_mlp_riscv_rvv.s

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
  wider_mlp_riscv_rvv.s \
  -march=rv64gcv \
  -mabi=lp64d \
  -fPIC \
  -o wider_mlp_riscv_rvv.o

# Generated:
#   wider_mlp_riscv_rvv.o

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
find . -maxdepth 1 -name '*.o' ! -name 'wider_mlp_riscv_rvv.o' -exec mv {} runtime_objs/ \;

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
  wider_mlp_riscv_rvv.o \
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
  wider_mlp_riscv_rvv.o \
  runtime_objs/*.o \
  -o wider_mlp_riscv_rvv \
  -I $ONNX_MLIR_INCLUDE \
  -march=rv64gcv \
  -mabi=lp64d \
  -static

echo "- Build complete: $BUILD_DIR/wider_mlp_riscv_rvv"
