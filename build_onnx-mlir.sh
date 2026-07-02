#!/bin/bash
#
# build_onnx_mlir.sh
# ------------------------------------------------------------
# Script for configuring, building, and testing ONNX-MLIR
# using an externally built LLVM/MLIR toolchain.
#
# This script:
#   1. Configures ONNX-MLIR with CMake + Ninja
#   2. Uses a custom LLVM/MLIR build
#   3. Supports virtual environments / custom Python installs
#   4. Builds ONNX-MLIR in Release mode
#   5. Runs ONNX-MLIR lit regression tests
#
# Requirements:
#   - LLVM + MLIR already built
#   - Ninja installed
#   - Python virtual environment activated
#   - ONNX-MLIR source cloned recursively
#
# Example:
#   git clone --recursive https://github.com/onnx/onnx-mlir.git
#
# Usage:
#   chmod +x build_onnx_mlir.sh
#   ./build_onnx_mlir.sh
#
# Optional:
#   export pythonLocation=/path/to/python/install
# ------------------------------------------------------------

# ============================================================
# Environment Cleanup
# ============================================================

# Unset compiler/linker environment variables to avoid
# interference from previously configured toolchains.
#
# This ensures CMake uses the explicitly provided compilers.
#
unset CC CXX LD

# ============================================================
# LLVM / MLIR Configuration
# ============================================================

# Path to MLIR CMake configuration files.
#
# Required by ONNX-MLIR to locate the MLIR installation.
#
MLIR_DIR=$(pwd)/llvm-project/build/lib/cmake/mlir

# Path to LLVM CMake configuration files.
#
# Required by ONNX-MLIR to locate LLVM libraries/tools.
#
LLVM_DIR=$(pwd)/llvm-project/build/lib/cmake/llvm

# ============================================================
# Build Directory Setup
# ============================================================

# Remove previous build directory if desired.
# Uncomment for a clean rebuild.
#
# rm -rf build

# Create ONNX-MLIR build directory and enter it.
#
mkdir onnx-mlir/build && cd onnx-mlir/build

# ============================================================
# Configure ONNX-MLIR with CMake
# ============================================================

# Check whether a custom Python installation path
# was provided through the "pythonLocation" variable.
#
# If not provided:
#   - Use the currently active virtual environment.
#
# If provided:
#   - Use the specified Python root directory.
#
if [[ -z "$pythonLocation" ]]; then

  # ----------------------------------------------------------
  # Standard virtual environment configuration
  # ----------------------------------------------------------

  # Use system C++ compiler.
  # Build optimized Release binaries.
  # Additional package search path for LLVM/MLIR.
  # Enable LLVM runtime assertions.
  # Limit parallel linker jobs to reduce RAM usage.
  # MLIR installation path.
  # LLVM installation path.
  # Python interpreter from active virtual environment.
  cmake -G Ninja \
    -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=/home/laura/onnx-mlir/llvm-project/build \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_PARALLEL_LINK_JOBS=1 \
    -DMLIR_DIR="${MLIR_DIR}" \
    -DLLVM_DIR="${LLVM_DIR}" \
    -DPython3_EXECUTABLE="$VIRTUAL_ENV/bin/python" \
    ..

else

  # ----------------------------------------------------------
  # Custom Python installation configuration
  # ----------------------------------------------------------

  # Use system C++ compiler.
  # Build optimized Release binaries.
  # Additional LLVM package search path.
  # Enable LLVM assertions.
  # Restrict linker parallelism.
  # Root directory for custom Python installation.
  # MLIR CMake package path.
  # LLVM CMake package path.
  # Python executable from active virtual environment.
  cmake -G Ninja \
    -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=/home/laura/onnx-mlir/llvm-project/build \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_PARALLEL_LINK_JOBS=1 \
    -DPython3_ROOT_DIR="$pythonLocation" \
    -DMLIR_DIR="${MLIR_DIR}" \
    -DLLVM_DIR="${LLVM_DIR}" \
    -DPython3_EXECUTABLE="$VIRTUAL_ENV/bin/python" \
    ..

fi
# ============================================================
# Build Configuration
# ============================================================

# Number of parallel compilation jobs.
#
# "-j6": use up to 6 concurrent build processes.
#
export MAKEFLAGS="-j6"

# ============================================================
# Build ONNX-MLIR
# ============================================================

# Compile the entire ONNX-MLIR project.
#
# Uses Ninja through CMake.
#
cmake --build . -- ${MAKEFLAGS}

# ============================================================
# Run Regression / Lit Tests
# ============================================================

# Enable verbose test output.
#
# Useful for debugging failed tests.
#
export LIT_OPTS="-v"

# Run ONNX-MLIR lit test suite.
#
# This validates:
#   - Frontend lowering
#   - MLIR transformations
#   - Runtime behavior
#   - Backend code generation
#
cmake --build . --target check-onnx-lit

# ============================================================
# Build Complete
# ============================================================

echo "- ONNX-MLIR build and tests completed successfully."
