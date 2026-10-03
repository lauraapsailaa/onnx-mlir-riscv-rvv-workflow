#!/bin/bash
set -e

# ============================================================
# Plain float32 (NO TAFFO) build of ResNet20 -- diagnostic build to
# isolate whether the near-uniform, near-random board output we've been
# seeing is caused by something in TAFFO's own precision-tuning pipeline,
# or by something else entirely (onnx-mlir's own lowering, cross-
# compilation, the runtime). Steps 1-3 are IDENTICAL to
# build_resnet20_taffo.sh; Steps 4-5 (TAFFO annotation/quantization) are
# skipped entirely, going straight from Step 3's already-lowered,
# already-valid float32 MLIR to Step 6's LLVM lowering.
# ============================================================

RESNET20_DIR=~/onnx-mlir/onnx-mlir/docs/other_examples/ResNet20-noBN-cifar10
BUILD_DIR=${BUILD_DIR_OVERRIDE:-$RESNET20_DIR/resnet20_plain_float}

mkdir -p $BUILD_DIR

ONNX_MODEL=${ONNX_MODEL_OVERRIDE:-$RESNET20_DIR/resnet20_cifar10.onnx}

ONNX_MLIR_OPT=~/onnx-mlir/onnx-mlir/build/Release/bin/onnx-mlir-opt

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVER_SRC=${DRIVER_SRC_OVERRIDE:-$SCRIPT_DIR/resnet20.cpp}

echo "=============================================="
echo "STEP 0 - Setup"
echo "=============================================="
echo "BUILD_DIR:  $BUILD_DIR"
echo "ONNX_MODEL: $ONNX_MODEL"
echo ""

# ============================================================
# Step 1 - Generate ONNX MLIR
# ============================================================

echo "=============================================="
echo "STEP 1 - Generate ONNX MLIR"
echo "=============================================="

onnx-mlir $ONNX_MODEL --EmitONNXBasic -o $BUILD_DIR/resnet20_initial

echo "Step 1 done -> $BUILD_DIR/resnet20_initial.onnx.mlir"
echo ""

# ============================================================
# Step 2 - Lower to Krnl/Affine/Arith
# ============================================================

echo "=============================================="
echo "STEP 2 - Lower to Krnl/Affine/Arith"
echo "=============================================="

$ONNX_MLIR_OPT \
    --shape-inference --canonicalize \
    --decompose-onnx \
    --convert-onnx-to-krnl --convert-krnl-to-affine \
    --reconcile-unrealized-casts --canonicalize --cse \
    $BUILD_DIR/resnet20_initial.onnx.mlir \
    -o $BUILD_DIR/resnet20_pre_taffo.mlir

echo "Step 2 done -> $BUILD_DIR/resnet20_pre_taffo.mlir"
echo ""

# ============================================================
# Step 3 - Fix up onnx.Return / tensor-cast-before-return
# ============================================================

echo "=============================================="
echo "STEP 3 - Patch onnx.Return and cast-before-return"
echo "=============================================="

sed 's/onnx.Return/func.return/g' \
  $BUILD_DIR/resnet20_pre_taffo.mlir \
  > $BUILD_DIR/resnet20_pre_taffo_fixed.mlir

RETURN_LINE=$(grep -n "func.return %\S\+ : tensor<[^>]\+>" $BUILD_DIR/resnet20_pre_taffo_fixed.mlir | tail -1)
RETURN_TENSOR_NAME=$(echo "$RETURN_LINE" | grep -oP 'func\.return \K%\S+')
RETURN_SHAPE=$(echo "$RETURN_LINE" | grep -oP ': tensor<\K[^>]+(?=>)')
if [ -z "$RETURN_TENSOR_NAME" ] || [ -z "$RETURN_SHAPE" ]; then
  echo "ERROR: could not find 'func.return %X : tensor<SHAPE>' -- inspect the file manually"
  exit 1
fi
echo "  Detected output shape: $RETURN_SHAPE"

CAST_LINE=$(grep -n "$RETURN_TENSOR_NAME = builtin.unrealized_conversion_cast %\S\+ : memref<${RETURN_SHAPE}> to tensor<${RETURN_SHAPE}>" \
  $BUILD_DIR/resnet20_pre_taffo_fixed.mlir)
RETURN_MEMREF_NAME=$(echo "$CAST_LINE" | grep -oP 'unrealized_conversion_cast \K%\S+')
if [ -z "$RETURN_MEMREF_NAME" ]; then
  echo "ERROR: could not find the cast defining $RETURN_TENSOR_NAME -- inspect the file manually"
  exit 1
fi

echo "  Return value: $RETURN_TENSOR_NAME (tensor), underlying buffer: $RETURN_MEMREF_NAME (memref)"

sed -i \
  -e "/${RETURN_TENSOR_NAME} = builtin.unrealized_conversion_cast ${RETURN_MEMREF_NAME} : memref<${RETURN_SHAPE}> to tensor<${RETURN_SHAPE}>/d" \
  -e "s|func.return ${RETURN_TENSOR_NAME} :|func.return ${RETURN_MEMREF_NAME} :|" \
  -e "s|: tensor<${RETURN_SHAPE}>\$|: memref<${RETURN_SHAPE}>|" \
  $BUILD_DIR/resnet20_pre_taffo_fixed.mlir

echo "Step 3 done -> $BUILD_DIR/resnet20_pre_taffo_fixed.mlir"
echo ""

# ============================================================
# Steps 4-5 SKIPPED ENTIRELY -- no TAFFO annotation, no
# raise-to-taffo/VRA/dt-optimization/lower-to-arith. The file is already
# valid, complete, pure-float32 krnl/affine/arith MLIR at this point.
# ============================================================

echo "=============================================="
echo "STEPS 4-5 SKIPPED (plain float32 build, no TAFFO)"
echo "=============================================="
echo ""

# ============================================================
# Step 6 - Lower to LLVM Dialect (directly from Step 3's output)
# ============================================================

echo "=============================================="
echo "STEP 6 - Lower to LLVM Dialect"
echo "=============================================="

# --lower-krnl-region still needed: Step 2's --convert-onnx-to-krnl
# includes our own Conv.cpp KrnlRegionOp fix, so krnl.region ops exist
# in this file regardless of whether TAFFO ever touches it.
$ONNX_MLIR_OPT --lower-krnl-region --convert-krnl-to-llvm --canonicalize --cse \
  $BUILD_DIR/resnet20_pre_taffo_fixed.mlir -o $BUILD_DIR/resnet20_llvm.mlir

echo "Step 6 done -> $BUILD_DIR/resnet20_llvm.mlir"
echo ""

# ============================================================
# Step 7 - MLIR -> LLVM IR
# ============================================================

echo "=============================================="
echo "STEP 7 - MLIR -> LLVM IR"
echo "=============================================="

mlir-translate --mlir-to-llvmir \
  $BUILD_DIR/resnet20_llvm.mlir -o $BUILD_DIR/resnet20_riscv_rvv_plain.ll

echo "Step 7 done -> $BUILD_DIR/resnet20_riscv_rvv_plain.ll"
echo ""

# ============================================================
# Step 7b - Run LLVM's real optimizer (enables the Loop Vectorizer)
# ============================================================

echo "=============================================="
echo "STEP 7b - opt -O3 (real LLVM vectorization pass)"
echo "=============================================="

opt -O3 -mtriple=riscv64 -mattr=+v,+d,+m \
  -S $BUILD_DIR/resnet20_riscv_rvv_plain.ll \
  -o $BUILD_DIR/resnet20_riscv_rvv_plain_opt.ll

echo "Step 7b done -> $BUILD_DIR/resnet20_riscv_rvv_plain_opt.ll"
echo ""

# ============================================================
# Step 8 - LLVM IR -> RISC-V Assembly
# ============================================================

echo "=============================================="
echo "STEP 8 - LLVM IR -> RISC-V Assembly"
echo "=============================================="

llc -march=riscv64 -mattr=+v,+d,+m -relocation-model=pic \
  $BUILD_DIR/resnet20_riscv_rvv_plain_opt.ll -o $BUILD_DIR/resnet20_riscv_rvv_plain.s

echo "Step 8 done -> $BUILD_DIR/resnet20_riscv_rvv_plain.s"
echo ""

# ============================================================
# Step 9 - Assemble
# ============================================================

echo "=============================================="
echo "STEP 9 - Assemble"
echo "=============================================="

riscv64-unknown-linux-gnu-gcc -c \
  $BUILD_DIR/resnet20_riscv_rvv_plain.s \
  -march=rv64gcv -mabi=lp64d -fPIC \
  -o $BUILD_DIR/resnet20_riscv_rvv_plain.o

echo "Step 9 done -> $BUILD_DIR/resnet20_riscv_rvv_plain.o"
echo ""

# ============================================================
# Step 10 - Compile ONNX-MLIR Runtime Sources
# ============================================================

echo "=============================================="
echo "STEP 10 - Compile ONNX-MLIR Runtime Sources"
echo "=============================================="

ONNX_MLIR_INCLUDE=~/onnx-mlir/onnx-mlir/include
RUNTIME_DIR=~/onnx-mlir/onnx-mlir/riscv_build/src/Runtime
SUPPORT_DIR=~/onnx-mlir/onnx-mlir/riscv_build/src/Support
RUNTIME_OBJ_DIR=$BUILD_DIR/runtime_objs
mkdir -p $RUNTIME_OBJ_DIR

riscv64-unknown-linux-gnu-gcc -c \
  $RUNTIME_DIR/*.c $SUPPORT_DIR/*.c \
  -I $ONNX_MLIR_INCLUDE -march=rv64gcv -mabi=lp64d -fPIC

find . -maxdepth 1 -name '*.o' ! -name 'resnet20_riscv_rvv_plain.o' -exec mv {} $RUNTIME_OBJ_DIR/ \;

echo "Step 10 done -> runtime object files in $RUNTIME_OBJ_DIR"
echo ""

# ============================================================
# Step 11 - Link Shared Library
# ============================================================

echo "=============================================="
echo "STEP 11 - Link Shared Library"
echo "=============================================="

riscv64-unknown-linux-gnu-gcc -shared \
  $BUILD_DIR/resnet20_riscv_rvv_plain.o \
  $RUNTIME_OBJ_DIR/*.o \
  -o $BUILD_DIR/resnet20_riscv_rvv_plain.so \
  -march=rv64gc -mabi=lp64d

echo "Step 11 done -> $BUILD_DIR/resnet20_riscv_rvv_plain.so"
echo ""

# ============================================================
# Step 12 - Build Standalone Test Executable
# ============================================================

echo "=============================================="
echo "STEP 12 - Build Standalone Test Executable"
echo "=============================================="

if [ ! -f "$DRIVER_SRC" ]; then
  echo "  SKIPPED: no driver source found at $DRIVER_SRC"
else
  riscv64-unknown-linux-gnu-g++ \
    --std=c++11 -O3 \
    $DRIVER_SRC \
    $BUILD_DIR/resnet20_riscv_rvv_plain.o \
    $RUNTIME_OBJ_DIR/*.o \
    -o $BUILD_DIR/resnet20_riscv_rvv_plain \
    -I $ONNX_MLIR_INCLUDE \
    -march=rv64gcv -mabi=lp64d -static

  echo "Step 12 done -> $BUILD_DIR/resnet20_riscv_rvv_plain (statically linked)"
fi
echo ""

echo "=============================================="
echo "ALL STEPS COMPLETE (plain float32, no TAFFO)"
echo "=============================================="
