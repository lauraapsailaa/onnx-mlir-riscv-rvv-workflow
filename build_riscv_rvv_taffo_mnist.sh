#!/bin/bash
set -e  # stop immediately if any command fails

# ============================================================
# Directories
# ============================================================

MNIST_DIR=~/onnx-mlir/onnx-mlir/docs/mnist_example
BUILD_DIR=$MNIST_DIR/mnist_taffo

mkdir -p $BUILD_DIR

ONNX_MODEL=$MNIST_DIR/mnist.onnx

ONNX_MLIR_OPT=~/onnx-mlir/onnx-mlir/build/Release/bin/onnx-mlir-opt
TAFFO_OPT=~/onnx-mlir/TAFFO-MLIR/build_debug/bin/taffo-opt

# ONNX-MLIR runtime sources and public headers, needed to link a final
# executable against the OMTensor/OMTensorList C runtime API.
ONNX_MLIR_INCLUDE=~/onnx-mlir/onnx-mlir/include
RUNTIME_DIR=~/onnx-mlir/onnx-mlir/riscv_build/src/Runtime
SUPPORT_DIR=~/onnx-mlir/onnx-mlir/riscv_build/src/Support

# Directory containing extract_ranges.py / insert_annotations.py / compute_layer_ranges.py.
# Defaults to the same directory as this script; override if needed.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENERAL_PYTHON_SCRIPTS_DIR=~/onnx-mlir/onnx-mlir/docs/python_scripts
EXTRACT_SCRIPT=$GENERAL_PYTHON_SCRIPTS_DIR/extract_ranges.py
INSERT_SCRIPT=$GENERAL_PYTHON_SCRIPTS_DIR/insert_annotations.py
COMPUTE_SCRIPT=$GENERAL_PYTHON_SCRIPTS_DIR/compute_layer_ranges.py
LOAD_PROFILED_SCRIPT=$GENERAL_PYTHON_SCRIPTS_DIR/load_profiled_ranges.py

# "static" (interval-arithmetic worst-case bounds) or "profiled" (empirical
# ranges observed on real test data, from profile_ranges.py). See
# compute_layer_ranges.py vs load_profiled_ranges.py.
RANGE_SOURCE=static

# Only used when RANGE_SOURCE=profiled:
PROFILED_RANGES_JSON=$BUILD_DIR/mnist_profiled_ranges.json
PROFILED_MAXPOOL_TENSOR="onnx::Reshape_5"
PROFILED_RELU_TENSOR="onnx::Gemm_9"
PROFILED_FC2_LOGITS_TENSOR="x.3"
PATCH_SCRIPT=$GENERAL_PYTHON_SCRIPTS_DIR/patch_raised_casts.py

# Path to a C/C++ driver source file that calls the model via the OMTensor
# runtime API (see e.g. onnx-mlir's own examples for the expected calling
# convention). Not provided by this script -- write your own or adapt one
# of onnx-mlir's sample drivers, then point this at it.
DRIVER_SRC=$SCRIPT_DIR/mnist.cpp

# Name of the model's input tensor, and the assumed input range/precision
# used for its manual annotation.
INPUT_TENSOR_NAME=image
# Profiled (empirically observed) input range -- NOT [0,1]. The real
# deployed input is normalized ((pixel/255 - 0.1307) / 0.3081, standard
# MNIST normalization), giving a range of roughly [-0.42, 2.82], confirmed
# by profiling real test images with profile_ranges.py. The earlier [0,1]
# assumption was WRONG and made every downstream hand-derived bound
# (MaxPool/ReLU/Softmax ranges) unsound for real inputs.
INPUT_MIN=-0.4242129623889923
INPUT_MAX=2.821486711502075
PRECISION=0.01

# Network architecture constants (MaxPool 2x2 -> Reshape -> fc1 -> ReLU -> fc2 -> Softmax).
# fc1_N / fc2_N are each layer's reduction dimension (input feature count).
FC1_WEIGHT_NAME=fc1.weight
FC1_BIAS_NAME=fc1.bias
FC1_N=196
FC2_WEIGHT_NAME=fc2.weight
FC2_BIAS_NAME=fc2.bias
FC2_N=128

# Additional memref SSA names that carry the same range as the input tensor
# but aren't caught by automatic reinterpret_cast alias tracing (e.g. a
# range-preserving MaxPool output buffer written to by a loop rather than
# derived directly from %arg0). Space-separated; add more as needed.
EXTRA_INPUT_ALIASES="%alloc"

# Line numbers (1-indexed, in mnist_pre_taffo_fixed.mlir) of the four ops that
# need manual range annotation beyond weights/biases/input: MaxPool's running
# max, the real ReLU after fc1, Softmax's max-reduction, and Softmax's exp.
# These are specific to this exact model/pipeline version -- the script
# verifies them below and aborts with a clear message if they've drifted.
LINE_MAXPOOL_MAXNUMF=23
LINE_RELU_MAXNUMF=57
LINE_SOFTMAX_MAXNUMF=84
LINE_SOFTMAX_EXP=90

echo "=============================================="
echo "STEP 0 - Setup"
echo "=============================================="
echo "BUILD_DIR:        $BUILD_DIR"
echo "ONNX_MODEL:       $ONNX_MODEL"
echo "ONNX_MLIR_OPT:    $ONNX_MLIR_OPT"
echo "TAFFO_OPT:        $TAFFO_OPT"
echo "EXTRACT_SCRIPT:   $EXTRACT_SCRIPT"
echo "INSERT_SCRIPT:    $INSERT_SCRIPT"
echo "COMPUTE_SCRIPT:   $COMPUTE_SCRIPT"
echo "INPUT_TENSOR_NAME: $INPUT_TENSOR_NAME  range=[$INPUT_MIN, $INPUT_MAX]  precision=$PRECISION"
echo ""

# ============================================================
# Step 1 - Generate ONNX MLIR
# ============================================================

echo "=============================================="
echo "STEP 1 - Generate ONNX MLIR"
echo "=============================================="

onnx-mlir \
    $ONNX_MODEL \
    --EmitONNXBasic \
    -o $BUILD_DIR/mnist_initial

echo "Step 1 done -> $BUILD_DIR/mnist_initial.onnx.mlir"
echo ""

# ============================================================
# Step 2 - Lower to Krnl/Affine/Arith
# ============================================================

echo "=============================================="
echo "STEP 2 - Lower to Krnl/Affine/Arith"
echo "=============================================="

$ONNX_MLIR_OPT \
    --shape-inference \
    --canonicalize \
    --convert-onnx-to-krnl \
    --convert-krnl-to-affine \
    --reconcile-unrealized-casts \
    --canonicalize \
    --cse \
    $BUILD_DIR/mnist_initial.onnx.mlir \
    -o $BUILD_DIR/mnist_pre_taffo.mlir

echo "Step 2 done -> $BUILD_DIR/mnist_pre_taffo.mlir"
echo ""

# ============================================================
# Step 3 - Fix up onnx.Return / tensor-cast-before-return
# ============================================================

echo "=============================================="
echo "STEP 3 - Patch onnx.Return and cast-before-return"
echo "=============================================="

echo "Step 3a - Patching onnx.Return -> func.return"
sed 's/onnx.Return/func.return/g' \
  $BUILD_DIR/mnist_pre_taffo.mlir \
  > $BUILD_DIR/mnist_pre_taffo_fixed.mlir

echo "Step 3b - Patching tensor-cast-before-return pattern (if present)"
CAST_LINE=$(grep -m1 'builtin.unrealized_conversion_cast' $BUILD_DIR/mnist_pre_taffo_fixed.mlir || true)
if [ -n "$CAST_LINE" ]; then
  echo "  Found cast line: $CAST_LINE"
  MEMREF_VAL=$(echo "$CAST_LINE" | sed -E 's/.*unrealized_conversion_cast (%[a-zA-Z0-9_]+) : (memref<[^>]+>).*/\1/')
  MEMREF_TYPE=$(echo "$CAST_LINE" | sed -E 's/.*unrealized_conversion_cast (%[a-zA-Z0-9_]+) : (memref<[^>]+>).*/\2/')
  TENSOR_VAL=$(echo "$CAST_LINE" | sed -E 's/^ *(%[a-zA-Z0-9_]+) = .*/\1/')
  echo "  Rewriting return to use $MEMREF_VAL : $MEMREF_TYPE"
  sed -i \
    -e "/builtin.unrealized_conversion_cast/d" \
    -e "s|func.return ${TENSOR_VAL} :|func.return ${MEMREF_VAL} :|" \
    -e "s|: tensor<[^>]*>\$|: ${MEMREF_TYPE}|" \
    $BUILD_DIR/mnist_pre_taffo_fixed.mlir
else
  echo "  No cast-before-return pattern found, skipping"
fi

echo "Step 3 fixed file ready -> $BUILD_DIR/mnist_pre_taffo_fixed.mlir"
echo ""

# ============================================================
# Step 4 - Manual annotation (weight/bias/input/activation ranges)
# ============================================================

echo "=============================================="
echo "STEP 4 - Extract ranges, compute layer bounds, and annotate"
echo "=============================================="

echo "Step 4a - Extracting exact weight/bias ranges from $ONNX_MODEL"
python3 $EXTRACT_SCRIPT \
  $ONNX_MODEL \
  --input-name $INPUT_TENSOR_NAME \
  --input-min $INPUT_MIN --input-max $INPUT_MAX \
  --precision $PRECISION \
  --json-out $BUILD_DIR/mnist_ranges.json

echo "Step 4a done -> $BUILD_DIR/mnist_ranges.json"
echo ""

echo "Step 4b - Computing derived layer ranges (post-ReLU, fc2 logits)"

if [ "$RANGE_SOURCE" = "profiled" ]; then
  if [ ! -f "$PROFILED_RANGES_JSON" ]; then
    echo "ERROR: RANGE_SOURCE=profiled but $PROFILED_RANGES_JSON does not exist."
    echo "Run profile_ranges.py first, e.g.:"
    echo "  python3 $SCRIPT_DIR/profile_ranges.py $ONNX_MODEL \\"
    echo "    --input-name $INPUT_TENSOR_NAME --num-images 200 \\"
    echo "    --json-out $PROFILED_RANGES_JSON"
    exit 1
  fi
  echo "  Using PROFILED (empirical) ranges from $PROFILED_RANGES_JSON"
  eval "$(python3 $LOAD_PROFILED_SCRIPT \
    $PROFILED_RANGES_JSON \
    --maxpool-tensor "$PROFILED_MAXPOOL_TENSOR" \
    --relu-tensor "$PROFILED_RELU_TENSOR" \
    --fc2-logits-tensor "$PROFILED_FC2_LOGITS_TENSOR")"
else
  echo "  Using STATIC (interval-arithmetic, worst-case) ranges"
  eval "$(python3 $COMPUTE_SCRIPT \
    $BUILD_DIR/mnist_ranges.json \
    --input-name $INPUT_TENSOR_NAME \
    --fc1-weight $FC1_WEIGHT_NAME --fc1-bias $FC1_BIAS_NAME --fc1-n $FC1_N \
    --fc2-weight $FC2_WEIGHT_NAME --fc2-bias $FC2_BIAS_NAME --fc2-n $FC2_N)"
fi

echo "  MaxPool range:      [$MAXPOOL_MIN, $MAXPOOL_MAX]"
echo "  ReLU range:         [$RELU_MIN, $RELU_MAX]"
echo "  Softmax max range:  [$SOFTMAX_MAX_MIN, $SOFTMAX_MAX_MAX]"
echo "  Softmax exp range:  [$SOFTMAX_EXP_MIN, $SOFTMAX_EXP_MAX]"
echo ""

# Named intermediate buffers that hold an already-computed value (e.g. a
# layer's activation output, stored via memref.store) which gets read back
# by a LATER op. The original value-level annotation doesn't survive a
# store/load round-trip through memory, since MLIR's dataflow analysis
# tracks SSA values, not memory contents -- so these loads need their own
# explicit range. Format: "SSA_NAME:MIN,MAX" entries, space-separated.
# %alloc_2 holds fc1's post-ReLU output, read back by fc2's matmul loop.
# %alloc_3 holds fc2's logits, read back by both softmax loops (the
# max-reduction and the sum-of-exponentials).
# %alloc_4 holds each softmax exp(x-max) value, stored during the
# sum-of-exponentials loop and read back by the final division loop.
BUFFER_RANGES="%alloc_2:${RELU_MIN},${RELU_MAX} %alloc_3:${SOFTMAX_MAX_MIN},${SOFTMAX_MAX_MAX} %alloc_4:${SOFTMAX_EXP_MIN},${SOFTMAX_EXP_MAX}"

echo "Step 4c - Verifying expected op line numbers still match"
verify_line() {
  local line_no=$1
  local expected_pattern=$2
  local actual=$(sed -n "${line_no}p" $BUILD_DIR/mnist_pre_taffo_fixed.mlir)
  if ! echo "$actual" | grep -q "$expected_pattern"; then
    echo "ERROR: line $line_no does not match expected pattern '$expected_pattern'"
    echo "  Actual content: $actual"
    echo "  The MLIR structure has likely changed (different onnx-mlir/model version)."
    echo "  Re-run: grep -n \"arith.maxnumf\\|math.exp\" $BUILD_DIR/mnist_pre_taffo_fixed.mlir"
    echo "  and update LINE_MAXPOOL_MAXNUMF / LINE_RELU_MAXNUMF / LINE_SOFTMAX_MAXNUMF / LINE_SOFTMAX_EXP accordingly."
    exit 1
  fi
  echo "  line $line_no OK: $actual"
}
verify_line $LINE_MAXPOOL_MAXNUMF "arith.maxnumf"
verify_line $LINE_RELU_MAXNUMF "arith.maxnumf"
verify_line $LINE_SOFTMAX_MAXNUMF "arith.maxnumf"
verify_line $LINE_SOFTMAX_EXP "math.exp"
echo "Step 4c done -- all expected op lines verified"
echo ""

echo "Step 4d - Inserting set_range annotations into the pre-raised MLIR"
EXTRA_ALIAS_ARGS=""
for alias in $EXTRA_INPUT_ALIASES; do
  EXTRA_ALIAS_ARGS="$EXTRA_ALIAS_ARGS --extra-input-alias $alias"
done

BUFFER_RANGE_ARGS=""
for spec in $BUFFER_RANGES; do
  BUFFER_RANGE_ARGS="$BUFFER_RANGE_ARGS --buffer-range $spec"
done

python3 $INSERT_SCRIPT \
  $BUILD_DIR/mnist_pre_taffo_fixed.mlir \
  $BUILD_DIR/mnist_ranges.json \
  --input-tensor-name $INPUT_TENSOR_NAME \
  $EXTRA_ALIAS_ARGS \
  $BUFFER_RANGE_ARGS \
  --op-range ${LINE_MAXPOOL_MAXNUMF}:${MAXPOOL_MIN},${MAXPOOL_MAX} \
  --op-range ${LINE_RELU_MAXNUMF}:${RELU_MIN},${RELU_MAX} \
  --op-range ${LINE_SOFTMAX_MAXNUMF}:${SOFTMAX_MAX_MIN},${SOFTMAX_MAX_MAX} \
  --op-range ${LINE_SOFTMAX_EXP}:${SOFTMAX_EXP_MIN},${SOFTMAX_EXP_MAX} \
  --accumulator-range=${SOFTMAX_MAX_MIN},${SOFTMAX_MAX_MAX} \
  -o $BUILD_DIR/mnist_pre_taffo_annotated.mlir

echo "Step 4d done -> $BUILD_DIR/mnist_pre_taffo_annotated.mlir"
echo ""


python3 ~/onnx-mlir/onnx-mlir/docs/python_scripts/fix_maxpool_init.py \
  $BUILD_DIR/mnist_pre_taffo_annotated.mlir \
  $BUILD_DIR/mnist_pre_taffo_annotated.mlir $MAXPOOL_MIN

# ============================================================
# Step 5 - TAFFO Precision Tuning
# ============================================================

echo "=============================================="
echo "STEP 5 - TAFFO Precision Tuning"
echo "=============================================="

echo "Step 5a - Raising to TAFFO dialect (consumes set_range annotations)"
$TAFFO_OPT \
  --allow-unregistered-dialect \
  --raise-to-taffo \
  $BUILD_DIR/mnist_pre_taffo_annotated.mlir \
  -o $BUILD_DIR/mnist_after_raise.mlir

echo "Step 5a done -> $BUILD_DIR/mnist_after_raise.mlir"

NUM_CAST2REAL=$(grep -c "taffo.cast2real" $BUILD_DIR/mnist_after_raise.mlir || true)
NUM_REMAINING_CALLS=$(grep -c "func.call" $BUILD_DIR/mnist_after_raise.mlir || true)
echo "  -> $NUM_CAST2REAL taffo.cast2real op(s) found (9 weights/biases+input+op-range overrides, plus one per accumulator load site)"
echo "  -> $NUM_REMAINING_CALLS remaining func.call(s) (should be 0)"
if [ "$NUM_REMAINING_CALLS" != "0" ]; then
  echo "  WARNING: some func.call ops were not consumed by raising -- check the annotated MLIR"
fi
echo ""

echo "Step 5b - Patching raise-generated unannotated loop-result casts"
# --raise-to-taffo itself introduces a fresh, unbounded
# 'builtin.unrealized_conversion_cast %X : f32 to !taffo.real' for the
# RESULT of certain reduction loops (Softmax's max-reduction and its
# sum-of-exponentials loop), with no counterpart in the pre-raised MLIR to
# annotate in advance. patch_raised_casts.py auto-detects these structurally
# (a cast whose operand is itself a loop's own f32 result) and reports each
# one keyed by its declaring loop's line number, which is stable across runs
# for a fixed model even though the cast's own SSA names can shift.
echo "  Discovering loop-result casts:"
python3 $PATCH_SCRIPT $BUILD_DIR/mnist_after_raise.mlir --list

# Known mapping for this model: the max-reduction loop is declared first
# (earlier line number) and needs fc2's logits range; the sum-of-exponentials
# loop is declared second and needs [1.0, 10.0] (each softmax exp term is in
# [0,1], and the max-element's own term is always exactly 1.0, so the 10-term
# sum is bounded in [1.0, 10.0]). If the model structure changes, re-run the
# --list command above and adjust the two `sed`-derived line numbers below.
LOOP_LINES=($(python3 $PATCH_SCRIPT $BUILD_DIR/mnist_after_raise.mlir --list 2>&1 \
  | sed -n '/problematic loop-RESULT cast/,/problematic loop-ENTRY/p' \
  | grep -oP 'declaring loop at line \K[0-9]+'))

if [ "${#LOOP_LINES[@]}" -ne 2 ]; then
  echo "  WARNING: expected exactly 2 problematic loop-result casts (softmax max "
  echo "  and softmax sum), found ${#LOOP_LINES[@]}. The network structure may have"
  echo "  changed. Proceeding with whatever was found, in declaration order --"
  echo "  double check the --list output above before trusting Step 5c."
fi

SOFTMAX_MAXLOOP_LINE=${LOOP_LINES[0]}
SOFTMAX_SUMLOOP_LINE=${LOOP_LINES[1]}
echo "  softmax max-reduction loop declared at line $SOFTMAX_MAXLOOP_LINE -> result range [$SOFTMAX_MAX_MIN, $SOFTMAX_MAX_MAX]"
echo "  softmax sum-of-exp loop declared at line $SOFTMAX_SUMLOOP_LINE -> result range [1.0, 10.0]"
echo "  softmax sum-of-exp loop's iter_arg accumulator -> entry range [0.0, 10.0] (starts at 0.0, unlike the tighter final-result bound)"

python3 $PATCH_SCRIPT \
  $BUILD_DIR/mnist_after_raise.mlir \
  --loop-range ${SOFTMAX_MAXLOOP_LINE}:${SOFTMAX_MAX_MIN},${SOFTMAX_MAX_MAX} \
  --loop-range ${SOFTMAX_SUMLOOP_LINE}:1.0,10.0 \
  --iterarg-range ${SOFTMAX_SUMLOOP_LINE}:0.0,10.0 \
  -o $BUILD_DIR/mnist_after_raise_patched.mlir

echo ""
echo "Step 5b done -> $BUILD_DIR/mnist_after_raise_patched.mlir"
echo ""

echo "Step 5c - Value range analysis + dt-optimization + lower-affine + lower-to-arith"
$TAFFO_OPT \
  --allow-unregistered-dialect \
  --value-range-analysis \
  --dt-optimization \
  --lower-affine --lower-to-arith --reconcile-unrealized-casts \
  $BUILD_DIR/mnist_after_raise_patched.mlir \
  -o $BUILD_DIR/mnist_taffo.mlir

echo "Step 5c done -> $BUILD_DIR/mnist_taffo.mlir"
echo ""

# ============================================================
# Step 6 - Lower to LLVM Dialect
# ============================================================

echo "=============================================="
echo "STEP 6 - Lower to LLVM Dialect"
echo "=============================================="

$ONNX_MLIR_OPT \
    --convert-krnl-to-llvm \
    --canonicalize \
    --cse \
    $BUILD_DIR/mnist_taffo.mlir \
    -o $BUILD_DIR/mnist_llvm.mlir

echo "Step 6 done -> $BUILD_DIR/mnist_llvm.mlir"
echo ""

# ============================================================
# Step 7 - MLIR -> LLVM IR
# ============================================================

echo "=============================================="
echo "STEP 7 - MLIR -> LLVM IR"
echo "=============================================="

mlir-translate \
    --mlir-to-llvmir \
    $BUILD_DIR/mnist_llvm.mlir \
    -o $BUILD_DIR/mnist_riscv_rvv_taffo.ll

echo "Step 7 done -> $BUILD_DIR/mnist_riscv_rvv_taffo.ll"
echo ""

# ============================================================
# Step 8 - LLVM IR -> RISC-V Assembly
# ============================================================

echo "=============================================="
echo "STEP 8 - LLVM IR -> RISC-V Assembly"
echo "=============================================="

opt -O3 -mtriple=riscv64 -mattr=+v,+d,+m \
    -S $BUILD_DIR/mnist_riscv_rvv_taffo.ll \
    -o $BUILD_DIR/mnist_riscv_rvv_taffo_opt.ll

llc \
    -march=riscv64 \
    -mattr=+v,+d,+m \
    -relocation-model=pic \
    $BUILD_DIR/mnist_riscv_rvv_taffo_opt.ll \
    -o $BUILD_DIR/mnist_riscv_rvv_taffo.s

echo "Step 8 done -> $BUILD_DIR/mnist_riscv_rvv_taffo.s"
echo ""

# ============================================================
# Step 9 - Assemble
# ============================================================

echo "=============================================="
echo "STEP 9 - Assemble"
echo "=============================================="

riscv64-unknown-linux-gnu-gcc \
    -c \
    $BUILD_DIR/mnist_riscv_rvv_taffo.s \
    -march=rv64gcv \
    -mabi=lp64d \
    -fPIC \
    -o $BUILD_DIR/mnist_riscv_rvv_taffo.o

echo "Step 9 done -> $BUILD_DIR/mnist_riscv_rvv_taffo.o"
echo ""

# ============================================================
# Step 10 - Compile ONNX-MLIR Runtime Sources
# ============================================================

echo "=============================================="
echo "STEP 10 - Compile ONNX-MLIR Runtime Sources"
echo "=============================================="

# Compile into a DEDICATED subdirectory, kept separate from BUILD_DIR's
# other contents, so Step 11 never accidentally links against stale .o
# files left over from earlier/unrelated runs of this script.
RUNTIME_OBJ_DIR=$BUILD_DIR/runtime_objs
mkdir -p $RUNTIME_OBJ_DIR

riscv64-unknown-linux-gnu-gcc -c \
  $RUNTIME_DIR/*.c \
  $SUPPORT_DIR/*.c \
  -I $ONNX_MLIR_INCLUDE \
  -march=rv64gcv \
  -mabi=lp64d \
  -fPIC

mv ./*.o $RUNTIME_OBJ_DIR/

echo "Step 10 done -> runtime object files in $RUNTIME_OBJ_DIR"
echo ""

# ============================================================
# Step 11 - Link Shared Library
# ============================================================

echo "=============================================="
echo "STEP 11 - Link Shared Library"
echo "=============================================="

# Link ONLY this run's model object plus this run's runtime objects --
# never a blind "$BUILD_DIR/*.o" glob, which would also pick up any stale
# .o files left over from earlier or unrelated runs (these export the same
# symbol names as the current model, causing "multiple definition" errors).
riscv64-unknown-linux-gnu-gcc -shared \
  $BUILD_DIR/mnist_riscv_rvv_taffo.o \
  $RUNTIME_OBJ_DIR/*.o \
  -o $BUILD_DIR/mnist_riscv_rvv_taffo.so \
  -march=rv64gc \
  -mabi=lp64d

echo "Step 11 done -> $BUILD_DIR/mnist_riscv_rvv_taffo.so"
echo ""

# ============================================================
# Step 12 - Build Standalone Test Executable
# ============================================================

echo "=============================================="
echo "STEP 12 - Build Standalone Test Executable"
echo "=============================================="

if [ ! -f "$DRIVER_SRC" ]; then
  echo "  SKIPPED: no driver source found at $DRIVER_SRC"
  echo "  Write a C/C++ driver that calls the model via the OMTensor runtime"
  echo "  API (create input/output OMTensors, run the model, read results),"
  echo "  save it as $DRIVER_SRC, and re-run this script to build the final"
  echo "  executable."
else
  # Link STATICALLY, directly against the object files -- NOT the .so.
  # A dynamically-linked executable bakes in the .so's absolute build-time
  # path, which the dynamic linker then looks for verbatim on the target
  # board (where it obviously doesn't exist, causing "cannot open shared
  # object file" at runtime). A single self-contained static binary has no
  # such runtime dependency and works regardless of the board's filesystem
  # layout.
  riscv64-unknown-linux-gnu-g++ \
    --std=c++11 -O3 \
    $DRIVER_SRC \
    $BUILD_DIR/mnist_riscv_rvv_taffo.o \
    $RUNTIME_OBJ_DIR/*.o \
    -o $BUILD_DIR/mnist_riscv_rvv_taffo \
    -I $ONNX_MLIR_INCLUDE \
    -march=rv64gcv \
    -mabi=lp64d \
    -static

  echo "Step 12 done -> $BUILD_DIR/mnist_riscv_rvv_taffo (statically linked)"
fi
echo ""

echo "=============================================="
echo "ALL STEPS COMPLETE"
echo "Object file:     $BUILD_DIR/mnist_riscv_rvv_taffo.o"
echo "Shared library:  $BUILD_DIR/mnist_riscv_rvv_taffo.so"
if [ -f "$BUILD_DIR/mnist_riscv_rvv_taffo" ]; then
  echo "Executable:      $BUILD_DIR/mnist_riscv_rvv_taffo  (copy to the board and run)"
fi
echo "=============================================="
