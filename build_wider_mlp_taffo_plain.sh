#!/bin/bash
set -e  # stop immediately if any command fails

# ============================================================
# Directories
# ============================================================

WIDER_MLP_DIR=~/onnx-mlir/onnx-mlir/docs/other_examples/wider_mlp
BUILD_DIR=$WIDER_MLP_DIR/wider_mlp_taffo_norvv

mkdir -p $BUILD_DIR

ONNX_MODEL=$WIDER_MLP_DIR/wider_mlp.onnx

ONNX_MLIR_OPT=~/onnx-mlir/onnx-mlir/build/Release/bin/onnx-mlir-opt
TAFFO_OPT=~/onnx-mlir/TAFFO-MLIR/build_debug/bin/taffo-opt

SCRIPT_DIR=~/onnx-mlir/onnx-mlir/docs/mnist_example
EXTRACT_SCRIPT=$SCRIPT_DIR/extract_ranges.py
INSERT_SCRIPT=$SCRIPT_DIR/insert_annotations.py
COMPUTE_SCRIPT=$SCRIPT_DIR/compute_layer_ranges_n.py
LOAD_PROFILED_SCRIPT=$SCRIPT_DIR/load_profiled_ranges_n.py
CLASSIFY_SCRIPT=$SCRIPT_DIR/classify_ops.py
PATCH_SCRIPT=$SCRIPT_DIR/patch_raised_casts.py
PROFILE_SCRIPT=$SCRIPT_DIR/profile_ranges.py

DRIVER_SRC=$WIDER_MLP_DIR/wider_mlp.cpp

# Model input: flat (1,784), plain [0,1] pixel values (no normalization,
# no MaxPool -- straight to the first Gemm).
INPUT_TENSOR_NAME=image
INPUT_MIN=0.0
INPUT_MAX=1.0
PRECISION=0.01

# Network architecture: 784 -> 1024 -> 512 -> 10, three dense layers.
FC1_WEIGHT_NAME=fc1.weight
FC1_BIAS_NAME=fc1.bias
FC1_N=784
FC2_WEIGHT_NAME=fc2.weight
FC2_BIAS_NAME=fc2.bias
FC2_N=1024
FC3_WEIGHT_NAME=fc3.weight
FC3_BIAS_NAME=fc3.bias
FC3_N=512
NUM_CLASSES=10

# "static" (interval-arithmetic worst-case, NOT recommended for this model
# -- static bounds were found to blow up to millions by the 3rd layer) or
# "profiled" (empirical ranges from profile_ranges.py). Default: profiled.
RANGE_SOURCE=profiled
PROFILED_RANGES_JSON=$BUILD_DIR/wider_mlp_profiled_ranges.json
# ONNX tensor names for each ReLU output and the final logits, in order --
# adjust these to match your actual train_wider_mlp.py export names
# (printed by that script's "Layer summary").
PROFILED_RELU_TENSORS=("relu1_out" "relu2_out")
PROFILED_LOGITS_TENSOR="gemm3_out"

# BUFFER_RANGES is set below, right after Step 4b computes RELU1_MIN/etc.
# (it references those values, so it can't be defined this early).

echo "=============================================="
echo "STEP 0 - Setup"
echo "=============================================="
echo "BUILD_DIR:      $BUILD_DIR"
echo "ONNX_MODEL:     $ONNX_MODEL"
echo "RANGE_SOURCE:   $RANGE_SOURCE"
echo ""

# ============================================================
# Step 1 - Generate ONNX MLIR
# ============================================================

echo "=============================================="
echo "STEP 1 - Generate ONNX MLIR"
echo "=============================================="

onnx-mlir $ONNX_MODEL --EmitONNXBasic -o $BUILD_DIR/wider_mlp_initial

echo "Step 1 done -> $BUILD_DIR/wider_mlp_initial.onnx.mlir"
echo ""

# ============================================================
# Step 2 - Lower to Krnl/Affine/Arith
# ============================================================

echo "=============================================="
echo "STEP 2 - Lower to Krnl/Affine/Arith"
echo "=============================================="

$ONNX_MLIR_OPT \
    --shape-inference --canonicalize \
    --convert-onnx-to-krnl --convert-krnl-to-affine \
    --reconcile-unrealized-casts --canonicalize --cse \
    $BUILD_DIR/wider_mlp_initial.onnx.mlir \
    -o $BUILD_DIR/wider_mlp_pre_taffo.mlir

echo "Step 2 done -> $BUILD_DIR/wider_mlp_pre_taffo.mlir"
echo ""

# ============================================================
# Step 3 - Fix up onnx.Return / tensor-cast-before-return
# ============================================================

echo "=============================================="
echo "STEP 3 - Patch onnx.Return and cast-before-return"
echo "=============================================="

sed 's/onnx.Return/func.return/g' \
  $BUILD_DIR/wider_mlp_pre_taffo.mlir \
  > $BUILD_DIR/wider_mlp_pre_taffo_fixed.mlir

CAST_LINE=$(grep -m1 'builtin.unrealized_conversion_cast' $BUILD_DIR/wider_mlp_pre_taffo_fixed.mlir || true)
if [ -n "$CAST_LINE" ]; then
  echo "  Found cast line: $CAST_LINE"
  MEMREF_VAL=$(echo "$CAST_LINE" | sed -E 's/.*unrealized_conversion_cast (%[a-zA-Z0-9_]+) : (memref<[^>]+>).*/\1/')
  MEMREF_TYPE=$(echo "$CAST_LINE" | sed -E 's/.*unrealized_conversion_cast (%[a-zA-Z0-9_]+) : (memref<[^>]+>).*/\2/')
  TENSOR_VAL=$(echo "$CAST_LINE" | sed -E 's/^ *(%[a-zA-Z0-9_]+) = .*/\1/')
  sed -i \
    -e "/builtin.unrealized_conversion_cast/d" \
    -e "s|func.return ${TENSOR_VAL} :|func.return ${MEMREF_VAL} :|" \
    -e "s|: tensor<[^>]*>\$|: ${MEMREF_TYPE}|" \
    $BUILD_DIR/wider_mlp_pre_taffo_fixed.mlir
else
  echo "  No cast-before-return pattern found, skipping"
fi

echo "Step 3 done -> $BUILD_DIR/wider_mlp_pre_taffo_fixed.mlir"
echo ""

# ============================================================
# Step 4 - Extract ranges, classify ops, and annotate
# ============================================================

echo "=============================================="
echo "STEP 4 - Extract ranges, classify ops, and annotate"
echo "=============================================="

echo "Step 4a - Extracting exact weight/bias ranges from $ONNX_MODEL"
python3 $EXTRACT_SCRIPT \
  $ONNX_MODEL \
  --input-name $INPUT_TENSOR_NAME \
  --input-min $INPUT_MIN --input-max $INPUT_MAX \
  --precision $PRECISION \
  --json-out $BUILD_DIR/wider_mlp_ranges.json

echo "Step 4a done -> $BUILD_DIR/wider_mlp_ranges.json"
echo ""

echo "Step 4b - Computing layer ranges (source: $RANGE_SOURCE)"
# Explicitly clear these first: if the branch below fails silently (e.g.
# eval "$(...)" evaluating an empty string when the inner python3 command
# errored -- which does NOT trigger set -e, since eval itself "succeeds"
# trivially on an empty string), any STALE value left over from an earlier
# run or branch in this same shell session must not be allowed to persist
# unnoticed. This turned into a real, hard-to-spot bug once already: a
# missing load_profiled_ranges_n.py silently left old static-computed
# values in place, with no visible error.
unset RELU1_MIN RELU1_MAX RELU2_MIN RELU2_MAX
unset SOFTMAX_MAX_MIN SOFTMAX_MAX_MAX SOFTMAX_EXP_MIN SOFTMAX_EXP_MAX
unset SOFTMAX_SUM_MIN SOFTMAX_SUM_MAX

if [ "$RANGE_SOURCE" = "profiled" ]; then
  if [ ! -f "$PROFILED_RANGES_JSON" ]; then
    echo "ERROR: RANGE_SOURCE=profiled but $PROFILED_RANGES_JSON does not exist."
    echo "Run profiling first, e.g.:"
    echo "  python3 $PROFILE_SCRIPT $ONNX_MODEL \\"
    echo "    --input-name $INPUT_TENSOR_NAME --num-images 200 --no-normalize --flatten \\"
    echo "    --json-out $PROFILED_RANGES_JSON"
    exit 1
  fi
  if [ ! -f "$LOAD_PROFILED_SCRIPT" ]; then
    echo "ERROR: LOAD_PROFILED_SCRIPT not found at $LOAD_PROFILED_SCRIPT"
    echo "Make sure load_profiled_ranges_n.py is present in \$SCRIPT_DIR ($SCRIPT_DIR)."
    exit 1
  fi
  RELU_TENSOR_ARGS=""
  for t in "${PROFILED_RELU_TENSORS[@]}"; do
    RELU_TENSOR_ARGS="$RELU_TENSOR_ARGS --relu-tensor $t"
  done
  eval "$(python3 $LOAD_PROFILED_SCRIPT \
    $PROFILED_RANGES_JSON \
    $RELU_TENSOR_ARGS \
    --logits-tensor $PROFILED_LOGITS_TENSOR \
    --num-classes $NUM_CLASSES)"
else
  if [ ! -f "$COMPUTE_SCRIPT" ]; then
    echo "ERROR: COMPUTE_SCRIPT not found at $COMPUTE_SCRIPT"
    exit 1
  fi
  eval "$(python3 $COMPUTE_SCRIPT \
    $BUILD_DIR/wider_mlp_ranges.json \
    --input-name $INPUT_TENSOR_NAME \
    --layer ${FC1_WEIGHT_NAME}:${FC1_BIAS_NAME}:${FC1_N} \
    --layer ${FC2_WEIGHT_NAME}:${FC2_BIAS_NAME}:${FC2_N} \
    --layer ${FC3_WEIGHT_NAME}:${FC3_BIAS_NAME}:${FC3_N} \
    --num-classes $NUM_CLASSES)"
fi

# Loud, immediate failure if the eval above didn't actually populate
# these -- instead of silently proceeding with unset/stale values.
for var in RELU1_MIN RELU1_MAX RELU2_MIN RELU2_MAX SOFTMAX_MAX_MIN SOFTMAX_MAX_MAX; do
  if [ -z "${!var}" ]; then
    echo "ERROR: $var was not set after Step 4b -- the range-computation"
    echo "command likely failed silently. Check the script paths above and"
    echo "try running the relevant python3 command manually to see the"
    echo "real error."
    exit 1
  fi
done

echo "  RELU1 range:        [$RELU1_MIN, $RELU1_MAX]"
echo "  RELU2 range:        [$RELU2_MIN, $RELU2_MAX]"
echo "  Softmax max range:  [$SOFTMAX_MAX_MIN, $SOFTMAX_MAX_MAX]"
echo "  Softmax exp range:  [$SOFTMAX_EXP_MIN, $SOFTMAX_EXP_MAX]"
echo "  Softmax sum range:  [$SOFTMAX_SUM_MIN, $SOFTMAX_SUM_MAX]"
echo ""

# Named intermediate memref buffers that hold an already-computed value
# (e.g. a layer's ReLU output, stored via memref.store) which gets read
# back by a LATER op. Confirmed via gdb + tracing crashing ops' operands
# back through wider_mlp_after_raise.mlir:
#   %alloc_1 (1x1024) = fc1's ReLU output, read back by fc2's matmul loop
#   %alloc_3 (1x512)  = fc2's ReLU output, read back by fc3's matmul loop
#   %alloc_4 (1x10)   = fc3's logits, read back by both softmax loops
#                       (the max-reduction and the sub/exp loop)
#   %alloc_5 (1x10)   = each softmax exp(x-max) value, read back by the
#                       final division loop
BUFFER_RANGES="%alloc_1:${RELU1_MIN},${RELU1_MAX} %alloc_3:${RELU2_MIN},${RELU2_MAX} %alloc_4:${SOFTMAX_MAX_MIN},${SOFTMAX_MAX_MAX} %alloc_5:${SOFTMAX_EXP_MIN},${SOFTMAX_EXP_MAX}"

echo "Step 4c - Classifying arith.maxnumf / math.exp occurrences"
python3 $CLASSIFY_SCRIPT $BUILD_DIR/wider_mlp_pre_taffo_fixed.mlir | tee $BUILD_DIR/classify_output.txt
echo ""

LINE_RELU1=$(grep -oP '^\s+line\s+\K[0-9]+(?=\s+relu_1\b)' $BUILD_DIR/classify_output.txt || true)
LINE_RELU2=$(grep -oP '^\s+line\s+\K[0-9]+(?=\s+relu_2\b)' $BUILD_DIR/classify_output.txt || true)
LINE_SOFTMAX_MAX=$(grep -oP '^\s+line\s+\K[0-9]+(?=\s+softmax_max\b)' $BUILD_DIR/classify_output.txt || true)
LINE_SOFTMAX_EXP=$(grep -oP '^\s+line\s+\K[0-9]+(?=\s+softmax_exp\b)' $BUILD_DIR/classify_output.txt || true)
UNCLASSIFIED_COUNT=$(grep -c "maxpool_or_unclassified" $BUILD_DIR/classify_output.txt || true)

echo "  relu_1 -> line $LINE_RELU1"
echo "  relu_2 -> line $LINE_RELU2"
echo "  softmax_max -> line $LINE_SOFTMAX_MAX"
echo "  softmax_exp -> line $LINE_SOFTMAX_EXP"
if [ "$UNCLASSIFIED_COUNT" != "0" ]; then
  echo "  WARNING: $UNCLASSIFIED_COUNT unclassified op(s) found -- check "
  echo "  $BUILD_DIR/classify_output.txt and inspect manually. This model "
  echo "  was expected to have NO MaxPool/unclassified ops."
fi
if [ -z "$LINE_RELU1" ] || [ -z "$LINE_RELU2" ] || [ -z "$LINE_SOFTMAX_MAX" ] || [ -z "$LINE_SOFTMAX_EXP" ]; then
  echo "ERROR: classify_ops.py did not find all four expected op categories."
  echo "Check $BUILD_DIR/classify_output.txt manually before proceeding."
  exit 1
fi
echo ""

echo "Step 4d - Inserting set_range annotations into the pre-raised MLIR"
BUFFER_RANGE_ARGS=""
for spec in $BUFFER_RANGES; do
  BUFFER_RANGE_ARGS="$BUFFER_RANGE_ARGS --buffer-range $spec"
done

python3 $INSERT_SCRIPT \
  $BUILD_DIR/wider_mlp_pre_taffo_fixed.mlir \
  $BUILD_DIR/wider_mlp_ranges.json \
  --input-tensor-name $INPUT_TENSOR_NAME \
  $BUFFER_RANGE_ARGS \
  --op-range ${LINE_RELU1}:${RELU1_MIN},${RELU1_MAX} \
  --op-range ${LINE_RELU2}:${RELU2_MIN},${RELU2_MAX} \
  --op-range ${LINE_SOFTMAX_MAX}:${SOFTMAX_MAX_MIN},${SOFTMAX_MAX_MAX} \
  --op-range ${LINE_SOFTMAX_EXP}:${SOFTMAX_EXP_MIN},${SOFTMAX_EXP_MAX} \
  --accumulator-range=${SOFTMAX_MAX_MIN},${SOFTMAX_MAX_MAX} \
  -o $BUILD_DIR/wider_mlp_pre_taffo_annotated.mlir

echo "Step 4d done -> $BUILD_DIR/wider_mlp_pre_taffo_annotated.mlir"
echo ""

# ============================================================
# Step 5 - TAFFO Precision Tuning
# ============================================================

echo "=============================================="
echo "STEP 5 - TAFFO Precision Tuning"
echo "=============================================="

echo "Step 5a - Raising to TAFFO dialect"
$TAFFO_OPT --allow-unregistered-dialect --raise-to-taffo \
  $BUILD_DIR/wider_mlp_pre_taffo_annotated.mlir \
  -o $BUILD_DIR/wider_mlp_after_raise.mlir

echo "Step 5a done -> $BUILD_DIR/wider_mlp_after_raise.mlir"
echo ""

echo "Step 5b - Patching raise-generated unannotated loop-result/entry casts"
python3 $PATCH_SCRIPT $BUILD_DIR/wider_mlp_after_raise.mlir --list

LOOP_LINES=($(python3 $PATCH_SCRIPT $BUILD_DIR/wider_mlp_after_raise.mlir --list 2>&1 \
  | sed -n '/problematic loop-RESULT cast/,/problematic loop-ENTRY/p' \
  | grep -oP 'declaring loop at line \K[0-9]+'))
ENTRY_LOOP_LINES=($(python3 $PATCH_SCRIPT $BUILD_DIR/wider_mlp_after_raise.mlir --list 2>&1 \
  | sed -n '/problematic loop-ENTRY/,$p' \
  | grep -oP 'declaring loop at line \K[0-9]+'))

PATCH_ARGS=""

if [ "${#LOOP_LINES[@]}" -ge 1 ]; then
  # Program order: the softmax max-reduction loop is declared first, the
  # sum-of-exponentials loop second (it depends on the max, computed
  # earlier) -- so LOOP_LINES[0] is the max loop's result cast, and any
  # further entries (LOOP_LINES[1], ...) are the sum loop's (and possibly
  # other loops', if the model structure ever grows) result casts.
  # Missing entries beyond index 0 was a real bug found on wider_mlp:
  # the sum loop's own result cast was silently left unpatched.
  SOFTMAX_MAXLOOP_LINE=${LOOP_LINES[0]}
  echo "  Loop-RESULT cast at declaring line $SOFTMAX_MAXLOOP_LINE (max-reduction) -> range [$SOFTMAX_MAX_MIN, $SOFTMAX_MAX_MAX]"
  PATCH_ARGS="$PATCH_ARGS --loop-range ${SOFTMAX_MAXLOOP_LINE}:${SOFTMAX_MAX_MIN},${SOFTMAX_MAX_MAX}"

  if [ "${#LOOP_LINES[@]}" -ge 2 ]; then
    SOFTMAX_SUMLOOP_RESULT_LINE=${LOOP_LINES[1]}
    echo "  Loop-RESULT cast at declaring line $SOFTMAX_SUMLOOP_RESULT_LINE (sum-of-exp) -> range [$SOFTMAX_SUM_MIN, $SOFTMAX_SUM_MAX]"
    PATCH_ARGS="$PATCH_ARGS --loop-range ${SOFTMAX_SUMLOOP_RESULT_LINE}:${SOFTMAX_SUM_MIN},${SOFTMAX_SUM_MAX}"
  fi
  if [ "${#LOOP_LINES[@]}" -ge 3 ]; then
    echo "  WARNING: ${#LOOP_LINES[@]} loop-RESULT casts found, only the first 2 are"
    echo "  handled automatically -- inspect the remaining ones manually:"
    printf '    line %s\n' "${LOOP_LINES[@]:2}"
  fi
fi

if [ "${#ENTRY_LOOP_LINES[@]}" -ge 1 ]; then
  # Only the softmax sum-of-exponentials loop's own iter_arg accumulator is
  # expected here. Its entry range is [0.0, N] (the accumulator genuinely
  # starts at 0.0) -- WIDER than the loop's own final-result bound
  # [1.0, N] used above, since the tighter bound only holds once the
  # max-element's term (always exactly 1.0) has actually been added.
  SOFTMAX_SUMLOOP_LINE=${ENTRY_LOOP_LINES[0]}
  echo "  Loop-ENTRY (iter_arg) cast at declaring line $SOFTMAX_SUMLOOP_LINE -> range [0.0, $SOFTMAX_SUM_MAX]"
  PATCH_ARGS="$PATCH_ARGS --iterarg-range ${SOFTMAX_SUMLOOP_LINE}:0.0,${SOFTMAX_SUM_MAX}"
fi

if [ -n "$PATCH_ARGS" ]; then
  python3 $PATCH_SCRIPT \
    $BUILD_DIR/wider_mlp_after_raise.mlir \
    $PATCH_ARGS \
    -o $BUILD_DIR/wider_mlp_after_raise_patched.mlir
else
  echo "  No loop-result or loop-entry casts found -- copying file unchanged"
  cp $BUILD_DIR/wider_mlp_after_raise.mlir $BUILD_DIR/wider_mlp_after_raise_patched.mlir
fi
echo ""

echo "Step 5c - Value range analysis + dt-optimization + lower-affine + lower-to-arith"
# --vra-mode=affine restricts value-range-analysis to the Affine path only,
# bypassing the Ntv (non-affine) path. TAFFO's default Mixed mode runs
# BOTH simultaneously and requires both to succeed -- but the Ntv path was
# found to have its own separate, unfixed bugs (crashing with "Division by
# zero detected" on a denominator that's actually [-inf,inf], suggesting
# it isn't consuming our set_range/cast2real annotations the same way the
# Affine path does). We've spent this whole session making the Affine
# path fully correct (early-abort fix, NaN-from-infinity fixes, exact
# corner-based division, etc.) -- vra-mode=affine lets us use that
# entirely, without needing to redo the same debugging for Ntv too.
#
# Output is redirected to a log file rather than the terminal: on a crash,
# taffo-opt's diagnostics dump the ENTIRE module as context, which for a
# model this size is very long. Only a short summary is shown here; the
# full log is available at $BUILD_DIR/step5c.log if you need to dig in
# (e.g. with gdb, or just reading the tail for the actual error message).
set +e
$TAFFO_OPT \
  --allow-unregistered-dialect --vra-mode=affine \
  --value-range-analysis --dt-optimization \
  --lower-affine --lower-to-arith --reconcile-unrealized-casts \
  $BUILD_DIR/wider_mlp_after_raise_patched.mlir \
  -o $BUILD_DIR/wider_mlp_taffo.mlir \
  > $BUILD_DIR/step5c.log 2>&1
STEP5C_EXIT=$?
set -e

if [ "$STEP5C_EXIT" -ne 0 ]; then
  echo "Step 5c FAILED (exit $STEP5C_EXIT)."
  echo "Relevant lines from $BUILD_DIR/step5c.log:"
  # A crash dump's actually-useful message (assertion text, or an MLIR
  # "error:"/"note:" diagnostic) appears near the TOP, followed by dozens
  # of boilerplate stack-frame lines -- so grep for the real content
  # instead of tailing (which only shows the unhelpful stack-frame tail).
  grep -n -m5 "Assertion\|error:\|note:" $BUILD_DIR/step5c.log || head -20 $BUILD_DIR/step5c.log
  echo ""
  echo "Full log (including full stack trace / module dump if present): $BUILD_DIR/step5c.log"
  exit 1
fi

echo "Step 5c done -> $BUILD_DIR/wider_mlp_taffo.mlir"

REMAINING_TAFFO=$(grep -c "taffo\." $BUILD_DIR/wider_mlp_taffo.mlir || true)
echo "  -> $REMAINING_TAFFO remaining 'taffo.' reference(s) (should be 0)"
echo ""

# ============================================================
# Step 6 - Lower to LLVM Dialect
# ============================================================

echo "=============================================="
echo "STEP 6 - Lower to LLVM Dialect"
echo "=============================================="

$ONNX_MLIR_OPT --convert-krnl-to-llvm --canonicalize --cse \
  $BUILD_DIR/wider_mlp_taffo.mlir -o $BUILD_DIR/wider_mlp_llvm.mlir

echo "Step 6 done -> $BUILD_DIR/wider_mlp_llvm.mlir"
echo ""

# ============================================================
# Step 7 - MLIR -> LLVM IR
# ============================================================

echo "=============================================="
echo "STEP 7 - MLIR -> LLVM IR"
echo "=============================================="

mlir-translate --mlir-to-llvmir \
  $BUILD_DIR/wider_mlp_llvm.mlir -o $BUILD_DIR/wider_mlp_riscv_taffo.ll

echo "Step 7 done -> $BUILD_DIR/wider_mlp_riscv_taffo.ll"
echo ""

# ============================================================
# Step 8 - LLVM IR -> RISC-V Assembly
# ============================================================

echo "=============================================="
echo "STEP 8 - LLVM IR -> RISC-V Assembly"
echo "=============================================="

llc -march=riscv64 -mattr=+d -relocation-model=pic \
  $BUILD_DIR/wider_mlp_riscv_taffo.ll -o $BUILD_DIR/wider_mlp_riscv_taffo.s

echo "Step 8 done -> $BUILD_DIR/wider_mlp_riscv_taffo.s"
echo ""

# ============================================================
# Step 9 - Assemble
# ============================================================

echo "=============================================="
echo "STEP 9 - Assemble"
echo "=============================================="

riscv64-unknown-linux-gnu-gcc -c \
  $BUILD_DIR/wider_mlp_riscv_taffo.s \
  -march=rv64gc -mabi=lp64d -fPIC \
  -o $BUILD_DIR/wider_mlp_riscv_taffo.o

echo "Step 9 done -> $BUILD_DIR/wider_mlp_riscv_taffo.o"
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
  -I $ONNX_MLIR_INCLUDE -march=rv64gc -mabi=lp64d -fPIC

mv ./*.o $RUNTIME_OBJ_DIR/

echo "Step 10 done -> runtime object files in $RUNTIME_OBJ_DIR"
echo ""

# ============================================================
# Step 11 - Link Shared Library
# ============================================================

echo "=============================================="
echo "STEP 11 - Link Shared Library"
echo "=============================================="

riscv64-unknown-linux-gnu-gcc -shared \
  $BUILD_DIR/wider_mlp_riscv_taffo.o \
  $RUNTIME_OBJ_DIR/*.o \
  -o $BUILD_DIR/wider_mlp_riscv_taffo.so \
  -march=rv64gc -mabi=lp64d

echo "Step 11 done -> $BUILD_DIR/wider_mlp_riscv_taffo.so"
echo ""

# ============================================================
# Step 12 - Build Standalone Test Executable
# ============================================================

echo "=============================================="
echo "STEP 12 - Build Standalone Test Executable"
echo "=============================================="

if [ ! -f "$DRIVER_SRC" ]; then
  echo "  SKIPPED: no driver source found at $DRIVER_SRC"
  echo "  Write a C/C++ driver (wider_mlp.cpp) matching mnist.cpp's structure"
  echo "  but with a flat (1,784) input tensor, save it there, and re-run."
else
  riscv64-unknown-linux-gnu-g++ \
    --std=c++11 -O3 \
    $DRIVER_SRC \
    $BUILD_DIR/wider_mlp_riscv_taffo.o \
    $RUNTIME_OBJ_DIR/*.o \
    -o $BUILD_DIR/wider_mlp_riscv_taffo \
    -I $ONNX_MLIR_INCLUDE \
    -march=rv64gc -mabi=lp64d -static

  echo "Step 12 done -> $BUILD_DIR/wider_mlp_riscv_taffo (statically linked)"
fi
echo ""

echo "=============================================="
echo "ALL STEPS COMPLETE"
echo "=============================================="
