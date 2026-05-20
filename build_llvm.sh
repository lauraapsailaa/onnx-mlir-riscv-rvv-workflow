#!/bin/bash
#
# build_llvm_mlir.sh
# ------------------------------------------------------------
# Script for downloading, configuring, and building
# LLVM + MLIR with RISC-V support.
#
# This setup is intended for ONNX-MLIR development and
# RISC-V backend experimentation.
#
# Enabled components:
#   - MLIR
#   - Clang
#   - OpenMP runtime
#   - RISC-V backend
#   - X86 backend
#
# Build system:
#   - CMake
#   - Ninja
#
# Final validation:
#   - Runs MLIR regression tests
#
# Requirements:
#   - git
#   - cmake
#   - ninja
#   - C++ compiler (gcc/g++)
#
# Example usage:
#   chmod +x build_llvm_mlir.sh
#   ./build_llvm_mlir.sh
# ------------------------------------------------------------

# Exit immediately if any command fails.
set -e

# ============================================================
# Step 1 - Clone LLVM Project
# ============================================================

# Clone the LLVM monorepo from GitHub.
#
# This repository includes:
#   - LLVM
#   - MLIR
#   - Clang
#   - OpenMP
#   - libc++
#   - lld
#   - and more
#
git clone https://github.com/llvm/llvm-project.git

# ============================================================
# Step 2 - Enter Repository
# ============================================================

cd llvm-project

# ============================================================
# Step 3 - Create Build Directory
# ============================================================

# Create an out-of-source build directory.
#
# Keeping build files separate from source files
# is recommended for LLVM development.
#
mkdir build

# Enter build directory.
cd build

# ============================================================
# Step 4 - Configure LLVM with CMake
# ============================================================

# Configure LLVM/MLIR build using Ninja generator.
#
# Main options:
#
#   -DLLVM_ENABLE_PROJECTS
#       Build selected LLVM subprojects:
#         * mlir
#         * clang
#
#   -DLLVM_ENABLE_RUNTIMES
#       Build runtime libraries:
#         * openmp
#
#   -DLLVM_TARGETS_TO_BUILD
#       Enable code generation backends:
#         * X86
#         * RISCV
#
#   -DCMAKE_BUILD_TYPE=Release
#       Build optimized binaries.
#
#   -DLLVM_ENABLE_ASSERTIONS=ON
#       Enable runtime/compiler assertions.
#       Useful for compiler debugging.
#
#   -DLLVM_ENABLE_RTTI=ON
#       Enable Run-Time Type Information (RTTI).
#       Required by several MLIR-based projects,
#       including ONNX-MLIR.
#
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS="mlir;clang" \
   -DLLVM_ENABLE_RUNTIMES="openmp" \
   -DLLVM_TARGETS_TO_BUILD="X86;RISCV" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DLLVM_ENABLE_RTTI=ON

# ============================================================
# Step 5 - Build and Run MLIR Tests
# ============================================================

# Build LLVM/MLIR and execute MLIR regression tests.
#
# "check-mlir" validates:
#   - MLIR infrastructure
#   - Dialects
#   - Passes
#   - Lowering pipelines
#   - Parser/printer correctness
#
# This step can take significant time depending
# on system performance.
#
cmake --build . --target check-mlir

# ============================================================
# Build Complete
# ============================================================

echo "- LLVM + MLIR build completed successfully."
