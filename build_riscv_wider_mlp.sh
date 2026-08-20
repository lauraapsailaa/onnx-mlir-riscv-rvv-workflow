#!/bin/bash
#
# build_riscv_wider_mlp.sh
# ------------------------------------------------------------
# Build script for compiling an ONNX WIDER_MLP model into a
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
#   4. Build a standalone test executable (statically linked).
#
# All generated files are written to a dedicated BUILD_DIR, kept separate
# from the RVV and TAFFO build scripts' own output directories.
#
# Result (inside BUILD_DIR):
#   - wider_mlp_riscv.so    (shared library)
#   - wider_mlp_riscv        (test executable)
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
#   chmod +x build_riscv_wider_mlp.sh
#   ./build_riscv_wider_mlp.sh
# ------------------------------------------------------------

# Exit immediately if any command fails.
# This prevents continuing with partial or invalid builds.
set -e

# ============================================================
# Configuration Section
# ============================================================

WIDER_MLP_DIR=~/onnx-mlir/onnx-mlir/docs/other_examples/wider_mlp
BUILD_DIR=$WIDER_MLP_DIR/wider_mlp_plain

mkdir -p $BUILD_DIR

# Path to the ONNX model to compile.
ONNX_MODEL=$WIDER_MLP_DIR/wider_mlp.onnx

# Path to ONNX-MLIR public header files.
ONNX_MLIR_INCLUDE=~/onnx-mlir/onnx-mlir/include

# Path to ONNX-MLIR runtime source files.
RUNTIME_DIR=~/onnx-mlir/onnx-mlir/riscv_build/src/Runtime

# Path to ONNX-MLIR support library source files.
SUPPORT_DIR=~/onnx-mlir/onnx-mlir/riscv_build/src/Support

# RISC-V GCC cross-compiler executable.
RISCV_GCC=~/riscv-gnu-toolchain/riscv/bin/riscv64-unknown-linux-gnu-gcc

# Name of the final shared library output.
OUTPUT=wider_mlp_riscv.so

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
#   -o wider_mlp_riscv
#       Output filename prefix.
#
#   -Xllc "-mattr=+d"
#       Enable the RISC-V double-precision floating-point extension.
#
onnx-mlir $ONNX_MODEL \
  --mtriple=riscv64-unknown-linux-gnu \
  -O3 \
  --EmitLLVMIR \
  -o wider_mlp_riscv \
  -Xllc "-mattr=+d,+m"

# Generated intermediate files may include:
#   wider_mlp_riscv.onnx.mlir

# ============================================================
# Step 1b - Translate MLIR to LLVM IR
# ============================================================

mlir-translate \
  --mlir-to-llvmir \
  wider_mlp_riscv.onnx.mlir \
  -o wider_mlp_riscv.ll

# ============================================================
# Step 1c - Run LLVM's real optimizer (enables the Loop Vectorizer)
# ============================================================

# llc alone does NOT run LLVM's middle-end optimization passes, including
# the Loop Vectorizer (confirmed empirically). The vectorizer only runs as
# part of opt's own pipeline -- run it explicitly here for a fair,
# consistent comparison against the RVV and TAFFO builds (which also now
# run this same step).
opt -O3 -mtriple=riscv64 -mattr=+d,+m \
  -S wider_mlp_riscv.ll \
  -o wider_mlp_riscv_opt.ll

# ============================================================
# Step 1d - Compile LLVM IR to RISC-V Assembly
# ============================================================

llc -march=riscv64 -mattr=+d,+m -relocation-model=pic \
  wider_mlp_riscv_opt.ll -o wider_mlp_riscv.s

$RISCV_GCC -c \
  wider_mlp_riscv.s \
  -march=rv64gc -mabi=lp64d -fPIC \
  -o wider_mlp_riscv.o

# Output generated:
#   wider_mlp_riscv.o

# ============================================================
# Step 2 - Compile ONNX-MLIR runtime sources
# ============================================================

echo "- Compiling runtime (C):"

# Compile runtime and support C source files into object files.
#
# Compiler options:
#   -c
#       Compile only (do not link yet).
#
#   -I
#       Add include directory for ONNX-MLIR headers.
#
#   -march=rv64gc
#       Target RISC-V ISA: rv64 = 64-bit RISC-V, g = general-purpose ISA
#       extensions, c = compressed instruction extension.
#
#   -mabi=lp64d
#       ABI using: lp64 = 64-bit long/pointer, d = double-precision
#       floating point ABI.
#
#   -fPIC
#       Generate position-independent code required for shared libraries.
#
mkdir -p runtime_objs
$RISCV_GCC -c \
  $RUNTIME_DIR/*.c \
  $SUPPORT_DIR/*.c \
  -I $ONNX_MLIR_INCLUDE \
  -march=rv64gc -mabi=lp64d \
  -fPIC
find . -maxdepth 1 -name '*.o' ! -name 'wider_mlp_riscv.o' -exec mv {} runtime_objs/ \;

# Example generated object files:
#   OMTensor.o
#   OMTensorList.o

# ============================================================
# Step 3 - Link shared library
# ============================================================

echo "- Linking shared library:"

# Link this run's model object plus this run's runtime objects only --
# never a blind "*.o" glob, which could also pick up unrelated .o files
# from the RVV/TAFFO builds if they ever shared a directory (these export
# the same symbol names, causing "multiple definition" link errors).
$RISCV_GCC -shared \
  wider_mlp_riscv.o \
  runtime_objs/*.o \
  -o $OUTPUT \
  -march=rv64gc -mabi=lp64d

# ============================================================
# Step 4 - Build Standalone Test Executable
# ============================================================

echo "- Building test executable:"

# Link STATICALLY, directly against the object files -- NOT the .so. A
# dynamically-linked executable bakes in the .so's absolute build-time
# path, which the dynamic linker then looks for verbatim on the target
# board (where it won't exist), causing "cannot open shared object file"
# at runtime. A single self-contained static binary avoids this entirely.
riscv64-unknown-linux-gnu-g++ \
  --std=c++11 -O3 \
  $DRIVER_SRC \
  wider_mlp_riscv.o \
  runtime_objs/*.o \
  -o wider_mlp_riscv \
  -I $ONNX_MLIR_INCLUDE \
  -march=rv64gc \
  -mabi=lp64d \
  -static

# ============================================================
# Build Complete
# ============================================================

echo "- Build complete: $BUILD_DIR/$OUTPUT, $BUILD_DIR/wider_mlp_riscv"
