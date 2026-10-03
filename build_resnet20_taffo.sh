#!/bin/bash
set -e

# ============================================================
# Directories
# ============================================================

RESNET20_DIR=~/onnx-mlir/onnx-mlir/docs/other_examples/ResNet20-noBN-cifar10
BUILD_DIR=${BUILD_DIR_OVERRIDE:-$RESNET20_DIR/resnet20_taffo}

mkdir -p $BUILD_DIR

ONNX_MODEL=${ONNX_MODEL_OVERRIDE:-$RESNET20_DIR/resnet20_cifar10.onnx}

ONNX_MLIR_OPT=~/onnx-mlir/onnx-mlir/build/Release/bin/onnx-mlir-opt
TAFFO_OPT=~/onnx-mlir/TAFFO-MLIR/build_debug/bin/taffo-opt

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENERAL_PYTHON_SCRIPTS_DIR=~/onnx-mlir/onnx-mlir/docs/python_scripts
EXTRACT_SCRIPT=$GENERAL_PYTHON_SCRIPTS_DIR/extract_ranges.py
INSERT_SCRIPT=$GENERAL_PYTHON_SCRIPTS_DIR/insert_annotations.py
CLASSIFY_SCRIPT=$GENERAL_PYTHON_SCRIPTS_DIR/classify_ops.py
PATCH_SCRIPT=$GENERAL_PYTHON_SCRIPTS_DIR/patch_raised_casts.py
PROFILE_SCRIPT=$GENERAL_PYTHON_SCRIPTS_DIR/profile_ranges.py
AGGREGATE_SCRIPT=$SCRIPT_DIR/compute_resnet20_aggregate_ranges.py

DRIVER_SRC=${DRIVER_SRC_OVERRIDE:-$SCRIPT_DIR/resnet20.cpp}

INPUT_TENSOR_NAME=image
INPUT_MIN=0.0
INPUT_MAX=1.0
PRECISION=0.0001
NUM_CLASSES=10

PROFILED_RANGES_JSON=$BUILD_DIR/resnet20_profiled_ranges.json

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
# Step 4 - Extract ranges, classify ops, and annotate
# ============================================================

echo "=============================================="
echo "STEP 4 - Extract ranges, classify ops, and annotate"
echo "=============================================="

echo "Step 4a - Extracting exact weight/bias ranges"
python3 $EXTRACT_SCRIPT \
  $ONNX_MODEL \
  --input-name $INPUT_TENSOR_NAME \
  --input-min $INPUT_MIN --input-max $INPUT_MAX \
  --precision $PRECISION \
  --json-out $BUILD_DIR/resnet20_ranges.json

echo "Step 4a done -> $BUILD_DIR/resnet20_ranges.json"
echo ""

echo "Step 4b - Computing aggregate (category-wide) profiled ranges"
if [ ! -f "$PROFILED_RANGES_JSON" ]; then
  echo "ERROR: $PROFILED_RANGES_JSON does not exist. Run profiling first:"
  echo "  python3 $PROFILE_SCRIPT $ONNX_MODEL --dataset cifar10 \\"
  echo "    --input-name $INPUT_TENSOR_NAME --num-images 200 --no-normalize \\"
  echo "    --json-out $PROFILED_RANGES_JSON"
  exit 1
fi

unset RELU_MIN RELU_MAX CONV_MIN CONV_MAX ADD_MIN ADD_MAX PAD_MIN PAD_MAX
unset GAP_MIN GAP_MAX LOGITS_MIN LOGITS_MAX ACCUMULATOR_MIN ACCUMULATOR_MAX
unset SOFTMAX_EXP_MIN SOFTMAX_EXP_MAX SOFTMAX_SUM_MIN SOFTMAX_SUM_MAX

eval "$(python3 $AGGREGATE_SCRIPT $PROFILED_RANGES_JSON --num-classes $NUM_CLASSES)"

for var in RELU_MIN RELU_MAX CONV_MIN CONV_MAX ADD_MIN ADD_MAX LOGITS_MIN LOGITS_MAX ACCUMULATOR_MIN ACCUMULATOR_MAX; do
  if [ -z "${!var}" ]; then
    echo "ERROR: $var was not set -- aggregation likely failed silently"
    exit 1
  fi
done

echo "  RELU range:        [$RELU_MIN, $RELU_MAX]"
echo "  Conv range:         [$CONV_MIN, $CONV_MAX]"
echo "  Add (skip) range:   [$ADD_MIN, $ADD_MAX]"
echo "  Logits range:       [$LOGITS_MIN, $LOGITS_MAX]"
echo "  Accumulator (catch-all) range: [$ACCUMULATOR_MIN, $ACCUMULATOR_MAX]"
# GlobalAveragePool's accumulator sums 64 spatial values BEFORE dividing
# by 64 -- its true range during accumulation is up to ~64x wider than
# the post-division GAP_MAX, and the same buffer is often reused for
# both the accumulation and post-division phases (a single --buffer-range
# applies uniformly to every load from that name), so this must be wide
# enough to safely cover both. Confirmed as the root cause of severe
# fixed-point saturation (values clipped at a hard ~0.5 ceiling) when the
# narrower GAP_MIN/GAP_MAX was used for the accumulator directly.
GAP_ACCUM_MAX=$(python3 -c "print(${GAP_MAX} * 64)")
echo "  GAP accumulator (pre-division) range: [$GAP_MIN, $GAP_ACCUM_MAX]"
echo ""

echo "Step 4c - Classifying arith.maxnumf / math.exp occurrences"
python3 $CLASSIFY_SCRIPT $BUILD_DIR/resnet20_pre_taffo_fixed.mlir | tee $BUILD_DIR/classify_output.txt
echo ""

RELU_LINES=($(grep -oP '^\s+line\s+\K[0-9]+(?=\s+relu_[0-9]+\b)' $BUILD_DIR/classify_output.txt || true))
LINE_SOFTMAX_MAX=$(grep -oP '^\s+line\s+\K[0-9]+(?=\s+softmax_max\b)' $BUILD_DIR/classify_output.txt || true)
LINE_SOFTMAX_EXP=$(grep -oP '^\s+line\s+\K[0-9]+(?=\s+softmax_exp\b)' $BUILD_DIR/classify_output.txt || true)
UNCLASSIFIED_COUNT=$(grep -c "maxpool_or_unclassified" $BUILD_DIR/classify_output.txt || true)

echo "  Found ${#RELU_LINES[@]} ReLU site(s)"
echo "  softmax_max -> line $LINE_SOFTMAX_MAX"
echo "  softmax_exp -> line $LINE_SOFTMAX_EXP"
if [ "$UNCLASSIFIED_COUNT" != "0" ]; then
  echo "  WARNING: $UNCLASSIFIED_COUNT unclassified op(s) found -- inspect"
  echo "  $BUILD_DIR/classify_output.txt manually."
fi
if [ -z "$LINE_SOFTMAX_MAX" ] || [ -z "$LINE_SOFTMAX_EXP" ] || [ "${#RELU_LINES[@]}" -eq 0 ]; then
  echo "ERROR: classify_ops.py did not find the expected op categories."
  exit 1
fi
echo ""

echo "Step 4d - Inserting set_range annotations"

RELU_MAP_SCRIPT=$SCRIPT_DIR/map_relu_ranges.py
RELU_OP_RANGE_ARGS=$(python3 $RELU_MAP_SCRIPT $PROFILED_RANGES_JSON "${RELU_LINES[@]}")
if [ $? -ne 0 ]; then
  echo "ERROR: map_relu_ranges.py failed -- see message above (likely a"
  echo "count mismatch between detected ReLU sites and profiled tensors)."
  exit 1
fi
OP_RANGE_ARGS="$RELU_OP_RANGE_ARGS"
OP_RANGE_ARGS="$OP_RANGE_ARGS --op-range ${LINE_SOFTMAX_MAX}:${LOGITS_MIN},${LOGITS_MAX}"
OP_RANGE_ARGS="$OP_RANGE_ARGS --op-range ${LINE_SOFTMAX_EXP}:${SOFTMAX_EXP_MIN},${SOFTMAX_EXP_MAX}"

# *** BUFFER_RANGES starts EMPTY -- discovered iteratively via crashes,
# same process used for every prior model. When Step 5c crashes, use gdb
# to trace the failing op's operand to an unannotated %allocN buffer,
# determine its category from the surrounding MLIR (Relu/Conv/Add store),
# add an entry here using the matching aggregate bound, and re-run.
# Skip-connection identity-path buffers, found via:
#   grep -B1 'builtin.unrealized_conversion_cast %107 : f32 to !taffo.real'
#     resnet20_after_raise_patched.mlir | grep -oP 'affine\.load \K%alloc_\d+' | sort -u
# One per residual block's Add (main path + identity path) -- these were
# completely unannotated (a bare unrealized_conversion_cast, no bounds at
# all), matching the "argRanges[1] inverted" crash exactly.
# %alloc_42: the flattened, post-GlobalAveragePool buffer (shape 1x64)
# feeding the final FC/Gemm layer's reduction loop -- found via the same
# gdb trace, unannotated (bare unrealized_conversion_cast). Uses the GAP
# aggregate bound, matching its actual source (GlobalAveragePool's output).
# %50, %53, ..., %104: all 19 Pad-output buffers (one per Conv with
# padding: 1 initial + 2 per residual block x 9 blocks). Found via the
# shared structural pattern "unrealized_conversion_cast %X : tensor<...>
# to memref<...>" -- insert_annotations.py's alias tracer only follows
# memref.reinterpret_cast chains, but Pad's own lowering produces this
# tensor-to-memref cast pattern instead, so none of these get picked up
# automatically. %50 is the FIRST one (conv1's own padded input, derived
# directly from the model's input image) -- uses the exact [0,1] input
# range. The remaining 18 are all derived from a prior ReLU output --
# use the aggregate PAD bound (a safe superset for any of them).
# %alloc_14, %alloc_27: inputs to the two downsample 1x1 convs
# (stage2.0, stage3.0) -- ReLU outputs feeding a downsample conv directly,
# bypassing Pad entirely since 1x1 kernels need no padding, which is why
# neither the earlier skip-connection-Add nor Pad-output searches caught
# them. %alloc_43: the logits buffer's SECOND use, inside Softmax's
# sum-of-exp loop (its first use, feeding arith.maxnumf directly on raw
# f32, is already correctly handled via --op-range).
# %alloc_44: the exp-value buffer, stored during Softmax's sum-of-exp
# loop and read back later for the division step -- same buffer
# read-back pattern seen for MNIST/wider_mlp's Softmax before.
# %alloc_5, %alloc_9, ..., %alloc_40: the main-path conv output buffers
# feeding each skip-connection Add -- the "other half" of the
# %alloc_2/6/10/etc identity-path buffers fixed earlier. Raw conv output,
# pre-Add, so uses CONV_MIN/CONV_MAX.
# *** RESET TO EMPTY after adding --decompose-onnx (Step 2): Pad now
# lowers to real krnl loops instead of an opaque tensor-to-memref bridge
# cast, which shifted the entire file's buffer numbering and made every
# previously-discovered %allocN name stale, and made the old %NN-style
# Pad-bridge entries entirely obsolete (that idiom no longer exists).
# Re-discovering iteratively via crashes, same process as before -- the
# CATEGORIES to expect are already known (skip-connection identity/
# main-path -> ADD/CONV, GlobalAveragePool output -> GAP, logits buffer,
# Softmax exp-value buffer, downsample-conv inputs -> RELU), just the
# specific %allocN names need re-finding.
# Found via find_unannotated_buffers.py in one pass (post --decompose-onnx,
# which fixed Pad's own lowering AND apparently let insert_annotations.py's
# existing alias-tracing auto-detect most of what previously needed manual
# discovery -- only 4 buffers remained unannotated this round, down from
# dozens before).
# %alloc_57: Pad's own memset+copy output buffer (init to constant, then
#   copy from the unpadded source into offset positions).
# %alloc_60: GlobalAveragePool's accumulator (init to 0, taffo.add loop,
#   then divide by 64).
# %alloc_62: the logits buffer (FC layer's bias-add result, pre-Softmax).
# %alloc_63: Softmax's exp-value buffer (stored during sum loop, re-read
#   during division).
# 28 more found via find_unannotated_buffers.py (scope-collision bug
# fixed), cleanly classified via pattern signal: Pad's offset-copy
# structure (17), ReLU's arith.maxnumf (9), skip-connection Add (2).
# %alloc (bare, no numeric suffix): conv1's own padded input buffer, the
# very first Pad in the network, derived directly from the input image --
# found via find_unannotated_buffers.py after fixing its regex to also
# match this bare-name case (it previously only matched %alloc_N).
BUFFER_MAP_SCRIPT=$SCRIPT_DIR/map_buffer_ranges.py
PAD_BUFFERS=(%alloc %alloc_4 %alloc_7 %alloc_10 %alloc_13 %alloc_16 %alloc_19 %alloc_22 %alloc_25 %alloc_29 %alloc_32 %alloc_35 %alloc_38 %alloc_41 %alloc_44 %alloc_48 %alloc_51 %alloc_54 %alloc_57)
PAD_BUFFER_RANGE_ARGS=$(python3 $BUFFER_MAP_SCRIPT $PROFILED_RANGES_JSON --pattern /Pad "${PAD_BUFFERS[@]}")
if [ $? -ne 0 ]; then
  echo "ERROR: map_buffer_ranges.py failed for Pad buffers"
  exit 1
fi
# Strip the "--buffer-range " prefixes back to bare "%name:min,max" specs,
# since BUFFER_RANGES (below) is later re-split into --buffer-range flags
# itself, and PAD_BUFFER_RANGE_ARGS already has them attached.
PAD_BUFFER_SPECS=$(echo "$PAD_BUFFER_RANGE_ARGS" | sed 's/--buffer-range //g')

BUFFER_RANGES="%alloc_60:${GAP_MIN},${GAP_ACCUM_MAX} %alloc_61:${GAP_MIN},${GAP_ACCUM_MAX} %alloc_62:${LOGITS_MIN},${LOGITS_MAX} %alloc_63:${SOFTMAX_EXP_MIN},${SOFTMAX_EXP_MAX} $PAD_BUFFER_SPECS %alloc_3:${RELU_MIN},${RELU_MAX} %alloc_9:${RELU_MIN},${RELU_MAX} %alloc_15:${RELU_MIN},${RELU_MAX} %alloc_21:${RELU_MIN},${RELU_MAX} %alloc_28:${RELU_MIN},${RELU_MAX} %alloc_34:${RELU_MIN},${RELU_MAX} %alloc_40:${RELU_MIN},${RELU_MAX} %alloc_47:${RELU_MIN},${RELU_MAX} %alloc_53:${RELU_MIN},${RELU_MAX} %alloc_27:${ADD_MIN},${ADD_MAX} %alloc_46:${ADD_MIN},${ADD_MAX}"

BUFFER_RANGE_ARGS=""
for spec in $BUFFER_RANGES; do
  BUFFER_RANGE_ARGS="$BUFFER_RANGE_ARGS --buffer-range $spec"
done

python3 $INSERT_SCRIPT \
  $BUILD_DIR/resnet20_pre_taffo_fixed.mlir \
  $BUILD_DIR/resnet20_ranges.json \
  --input-tensor-name $INPUT_TENSOR_NAME \
  --precision=$PRECISION \
  $BUFFER_RANGE_ARGS \
  $OP_RANGE_ARGS \
  --accumulator-range=${ACCUMULATOR_MIN},${ACCUMULATOR_MAX} \
  -o $BUILD_DIR/resnet20_pre_taffo_annotated.mlir

echo "Step 4d done -> $BUILD_DIR/resnet20_pre_taffo_annotated.mlir"
echo ""

# ============================================================
# Step 5 - TAFFO Precision Tuning
# ============================================================

echo "=============================================="
echo "STEP 5 - TAFFO Precision Tuning"
echo "=============================================="

echo "Step 5a - Raising to TAFFO dialect"
$TAFFO_OPT --allow-unregistered-dialect --raise-to-taffo \
  $BUILD_DIR/resnet20_pre_taffo_annotated.mlir \
  -o $BUILD_DIR/resnet20_after_raise.mlir

echo "Step 5a done -> $BUILD_DIR/resnet20_after_raise.mlir"
echo ""

echo "Step 5b - Patching raise-generated unannotated loop-result/entry casts"
python3 $PATCH_SCRIPT $BUILD_DIR/resnet20_after_raise.mlir --list

LOOP_LINES_RAW=($(python3 $PATCH_SCRIPT $BUILD_DIR/resnet20_after_raise.mlir --list 2>&1 \
  | sed -n '/problematic loop-RESULT cast/,/problematic loop-ENTRY/p' \
  | grep -oP 'declaring loop at line \K[0-9]+' | awk '!seen[$0]++'))
# Keep only OUTERMOST declaring lines (via genuine nesting/brace-depth,
# not trip count -- trip count alone is unreliable, since conv1's own
# outermost loop has trip count 3 too, from its 3-channel RGB input,
# same as the nested kh/kw levels of every OTHER conv layer). A conv
# layer's 3x3 reduction has 3 nested affine.for levels;
# patch_raised_casts.py's detection matches all of them, not just the
# one that actually corresponds to one distinct conv layer.
DEDUPE_SCRIPT=$SCRIPT_DIR/dedupe_nested_loops.py
LOOP_LINES=($(python3 $DEDUPE_SCRIPT $BUILD_DIR/resnet20_after_raise.mlir "${LOOP_LINES_RAW[@]}"))

ENTRY_LOOP_LINES=($(python3 $PATCH_SCRIPT $BUILD_DIR/resnet20_after_raise.mlir --list 2>&1 \
  | sed -n '/problematic loop-ENTRY/,$p' \
  | grep -oP 'declaring loop at line \K[0-9]+' | awk '!seen[$0]++'))

PATCH_ARGS=""

CONV_MAP_SCRIPT=$SCRIPT_DIR/map_conv_accumulator_ranges.py

NUM_LOOP_LINES=${#LOOP_LINES[@]}
if [ "$NUM_LOOP_LINES" -ge 1 ]; then
  # The last 2 entries (if present) are Softmax's max-reduction and
  # sum-of-exp loops (textually last in the file), NOT conv layers --
  # exclude them from the conv-accumulator mapping, they get their own
  # specific ranges below.
  if [ "$NUM_LOOP_LINES" -ge 2 ]; then
    CONV_LOOP_LINES=("${LOOP_LINES[@]:0:$((NUM_LOOP_LINES - 2))}")
  else
    CONV_LOOP_LINES=("${LOOP_LINES[@]}")
  fi

  if [ "${#CONV_LOOP_LINES[@]}" -ge 1 ]; then
    CONV_LOOP_RANGE_ARGS=$(python3 $CONV_MAP_SCRIPT \
      $PROFILED_RANGES_JSON $BUILD_DIR/resnet20_ranges.json \
      --flag=--loop-range \
      "${CONV_LOOP_LINES[@]}")
    if [ $? -ne 0 ]; then
      echo "ERROR: map_conv_accumulator_ranges.py failed for loop-RESULT casts"
      exit 1
    fi
    PATCH_ARGS="$PATCH_ARGS $CONV_LOOP_RANGE_ARGS"
    echo "  Applied precise per-layer ranges to ${#CONV_LOOP_LINES[@]} conv loop-RESULT cast(s)"
  fi

  if [ "$NUM_LOOP_LINES" -ge 2 ]; then
    # Softmax is textually last in the file, so its two loops (max-reduction,
    # sum-of-exp) are always the LAST two entries -- override them with
    # their specific, tighter ranges instead of the generic accumulator one.
    MAXLOOP_IDX=$((NUM_LOOP_LINES - 2))
    SUMLOOP_IDX=$((NUM_LOOP_LINES - 1))
    SOFTMAX_MAXLOOP_LINE=${LOOP_LINES[$MAXLOOP_IDX]}
    SOFTMAX_SUMLOOP_RESULT_LINE=${LOOP_LINES[$SUMLOOP_IDX]}
    echo "  Overriding last 2 (Softmax): line $SOFTMAX_MAXLOOP_LINE (max) -> [$LOGITS_MIN, $LOGITS_MAX], line $SOFTMAX_SUMLOOP_RESULT_LINE (sum) -> [$SOFTMAX_SUM_MIN, $SOFTMAX_SUM_MAX]"
    PATCH_ARGS="$PATCH_ARGS --loop-range ${SOFTMAX_MAXLOOP_LINE}:${LOGITS_MIN},${LOGITS_MAX}"
    PATCH_ARGS="$PATCH_ARGS --loop-range ${SOFTMAX_SUMLOOP_RESULT_LINE}:${SOFTMAX_SUM_MIN},${SOFTMAX_SUM_MAX}"
  fi
fi

if [ "${#ENTRY_LOOP_LINES[@]}" -ge 1 ]; then
  # Softmax's own sum-loop entry is textually LAST in the file (its 21
  # conv-layer %arg11 siblings all come before it), so it's the last
  # item here, not the first -- patch_raised_casts.py's own --list
  # detection catches all 22 iter_arg loops (21 conv + 1 softmax)
  # together, with no distinction between them at that stage.
  LAST_IDX=$((${#ENTRY_LOOP_LINES[@]} - 1))
  SOFTMAX_SUMLOOP_LINE=${ENTRY_LOOP_LINES[$LAST_IDX]}
  echo "  Loop-ENTRY (iter_arg) cast at declaring line $SOFTMAX_SUMLOOP_LINE -> range [0.0, $SOFTMAX_SUM_MAX]"
  PATCH_ARGS="$PATCH_ARGS --iterarg-range ${SOFTMAX_SUMLOOP_LINE}:0.0,${SOFTMAX_SUM_MAX}"
fi

# Conv-layer accumulator entry casts: one per conv's reduction loop
# (iter_arg-based, NOT the memref-based idiom RewriteMemAccumulator
# targets -- this is TAFFO-MLIR's own affine.for iter_args reduction).
# patch_raised_casts.py's own --list detection (built around Softmax's
# specific structure) doesn't recognize this pattern at all -- found
# instead via direct grep for the declaring affine.for line immediately
# preceding an unannotated "unrealized_conversion_cast %argN : f32 to
# !taffo.real" entry cast. Uses the same ACCUMULATOR catch-all bound as
# every other scalar accumulator in this pipeline.
# Computed dynamically, NOT hardcoded: the raised file's line numbers
# shift every time an upstream annotation changes (since that changes
# insert_annotations.py's output, which changes raise-to-taffo's output
# structure too) -- a static snapshot goes stale the moment any other
# BUFFER_RANGES/OP_RANGE entry is added or changed.
CONV_ITERARG_LINES=$(grep -n -B1 'builtin.unrealized_conversion_cast %arg11 : f32 to !taffo.real' \
  $BUILD_DIR/resnet20_after_raise.mlir \
  | grep 'affine.for' | grep -oP '^\d+')
CONV_ITERARG_LINES_ARR=($CONV_ITERARG_LINES)
if [ "${#CONV_ITERARG_LINES_ARR[@]}" -ge 1 ]; then
  CONV_ITERARG_RANGE_ARGS=$(python3 $CONV_MAP_SCRIPT \
    $PROFILED_RANGES_JSON $BUILD_DIR/resnet20_ranges.json \
    --flag=--iterarg-range \
    "${CONV_ITERARG_LINES_ARR[@]}")
  if [ $? -ne 0 ]; then
    echo "ERROR: map_conv_accumulator_ranges.py failed for loop-ENTRY casts"
    exit 1
  fi
  PATCH_ARGS="$PATCH_ARGS $CONV_ITERARG_RANGE_ARGS"
  echo "  Applied precise per-layer ranges to ${#CONV_ITERARG_LINES_ARR[@]} conv-layer accumulator entries"
fi

if [ -n "$PATCH_ARGS" ]; then
  python3 $PATCH_SCRIPT \
    $BUILD_DIR/resnet20_after_raise.mlir \
    --precision $PRECISION \
    $PATCH_ARGS \
    -o $BUILD_DIR/resnet20_after_raise_patched.mlir
else
  echo "  No loop-result or loop-entry casts found -- copying file unchanged"
  cp $BUILD_DIR/resnet20_after_raise.mlir $BUILD_DIR/resnet20_after_raise_patched.mlir
fi
echo ""

echo "Step 5b2 - Patching GlobalAveragePool's divisor constant"
# raise-to-taffo automatically wraps any never-annotated f32 value feeding
# a raised consumer in a generic, unbounded unrealized_conversion_cast.
# GlobalAveragePool's own "divide by pool size" constant (exactly 64.0,
# the 8x8 spatial pool size -- a genuine compile-time constant, not
# runtime data) hits this. Since insert_annotations.py has no mechanism
# for annotating plain arith.constant values feeding taffo.div directly
# (only memref/affine loads), this is patched here instead, at the same
# post-raise/pre-VRA stage patch_raised_casts.py already operates at.
# Confirmed via: grep -n '%cst = arith.constant' -> 6.400000e+01 : f32
sed -i -E \
  -e 's|(%[[:alnum:]_]+) = builtin.unrealized_conversion_cast %cst : f32 to !taffo.real|\1 = taffo.cast2real %cst, 0.01, 6.400000e+01, 6.400000e+01 : f32 -> !taffo.real|' \
  $BUILD_DIR/resnet20_after_raise_patched.mlir
echo "  Patched GlobalAveragePool divisor (%cst = 64.0) with exact bounds"
echo ""

echo "Step 5c - Value range analysis + dt-optimization + lower-affine + lower-to-arith"
set +e
$TAFFO_OPT \
  --allow-unregistered-dialect --vra-mode=affine \
  --value-range-analysis --dt-optimization \
  --lower-affine --lower-to-arith --reconcile-unrealized-casts \
  $BUILD_DIR/resnet20_after_raise_patched.mlir \
  -o $BUILD_DIR/resnet20_taffo.mlir \
  > $BUILD_DIR/step5c.log 2>&1
STEP5C_EXIT=$?
set -e

if [ "$STEP5C_EXIT" -ne 0 ]; then
  echo "Step 5c FAILED (exit $STEP5C_EXIT)."
  echo "Relevant lines from $BUILD_DIR/step5c.log:"
  grep -n -m5 "Assertion\|error:\|note:" $BUILD_DIR/step5c.log || head -20 $BUILD_DIR/step5c.log
  echo ""
  echo "Full log: $BUILD_DIR/step5c.log"
  echo ""
  echo "Next: gdb \$TAFFO_OPT with the same args, trace the failing op's"
  echo "operand to an unannotated %allocN buffer, add it to BUFFER_RANGES."
  exit 1
fi

echo "Step 5c done -> $BUILD_DIR/resnet20_taffo.mlir"

REMAINING_TAFFO=$(grep -c "taffo\." $BUILD_DIR/resnet20_taffo.mlir || true)
echo "  -> $REMAINING_TAFFO remaining 'taffo.' reference(s) (should be 0)"
echo ""

echo "Step 5d - Integer range narrowing (shrink i32 arith ops to i8/i16 where their"
echo "  computed range allows, for RVV vector-packing density)"
set +e
$TAFFO_OPT \
  --allow-unregistered-dialect \
  --pass-pipeline='builtin.module(arith-int-range-narrowing{int-bitwidths-supported=8,16,32})' \
  $BUILD_DIR/resnet20_taffo.mlir \
  -o $BUILD_DIR/resnet20_taffo_narrowed.mlir \
  > $BUILD_DIR/step5d.log 2>&1
STEP5D_EXIT=$?
set -e

if [ "$STEP5D_EXIT" -ne 0 ]; then
  echo "Step 5d FAILED (exit $STEP5D_EXIT)."
  echo "Relevant lines from $BUILD_DIR/step5d.log:"
  grep -n -m5 "Assertion\|error:\|note:" $BUILD_DIR/step5d.log || head -20 $BUILD_DIR/step5d.log
  echo ""
  echo "Full log: $BUILD_DIR/step5d.log"
  exit 1
fi

mv $BUILD_DIR/resnet20_taffo_narrowed.mlir $BUILD_DIR/resnet20_taffo.mlir
echo "Step 5d done -> $BUILD_DIR/resnet20_taffo.mlir (overwritten in place)"
echo ""

# ============================================================
# Step 6 - Lower to LLVM Dialect
# ============================================================

echo "=============================================="
echo "STEP 6 - Lower to LLVM Dialect"
echo "=============================================="

$ONNX_MLIR_OPT --lower-krnl-region --convert-krnl-to-llvm --canonicalize --cse \
  $BUILD_DIR/resnet20_taffo.mlir -o $BUILD_DIR/resnet20_llvm.mlir

echo "Step 6 done -> $BUILD_DIR/resnet20_llvm.mlir"
echo ""

# ============================================================
# Step 7 - MLIR -> LLVM IR
# ============================================================

echo "=============================================="
echo "STEP 7 - MLIR -> LLVM IR"
echo "=============================================="

mlir-translate --mlir-to-llvmir \
  $BUILD_DIR/resnet20_llvm.mlir -o $BUILD_DIR/resnet20_riscv_rvv_taffo.ll

echo "Step 7 done -> $BUILD_DIR/resnet20_riscv_rvv_taffo.ll"
echo ""

# ============================================================
# Step 7b - Run LLVM's real optimizer (enables the Loop Vectorizer)
# ============================================================

echo "=============================================="
echo "STEP 7b - opt -O3 (real LLVM vectorization pass)"
echo "=============================================="

opt -O3 -mtriple=riscv64 -mattr=+v,+d,+m \
  -S $BUILD_DIR/resnet20_riscv_rvv_taffo.ll \
  -o $BUILD_DIR/resnet20_riscv_rvv_taffo_opt.ll

echo "Step 7b done -> $BUILD_DIR/resnet20_riscv_rvv_taffo_opt.ll"
echo ""

# ============================================================
# Step 8 - LLVM IR -> RISC-V Assembly
# ============================================================

echo "=============================================="
echo "STEP 8 - LLVM IR -> RISC-V Assembly"
echo "=============================================="

llc -march=riscv64 -mattr=+v,+d,+m -relocation-model=pic \
  $BUILD_DIR/resnet20_riscv_rvv_taffo_opt.ll -o $BUILD_DIR/resnet20_riscv_rvv_taffo.s

echo "Step 8 done -> $BUILD_DIR/resnet20_riscv_rvv_taffo.s"
echo ""

# ============================================================
# Step 9 - Assemble
# ============================================================

echo "=============================================="
echo "STEP 9 - Assemble"
echo "=============================================="

riscv64-unknown-linux-gnu-gcc -c \
  $BUILD_DIR/resnet20_riscv_rvv_taffo.s \
  -march=rv64gcv -mabi=lp64d -fPIC \
  -o $BUILD_DIR/resnet20_riscv_rvv_taffo.o

echo "Step 9 done -> $BUILD_DIR/resnet20_riscv_rvv_taffo.o"
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

find . -maxdepth 1 -name '*.o' ! -name 'resnet20_riscv_rvv_taffo.o' -exec mv {} $RUNTIME_OBJ_DIR/ \;

echo "Step 10 done -> runtime object files in $RUNTIME_OBJ_DIR"
echo ""

# ============================================================
# Step 11 - Link Shared Library
# ============================================================

echo "=============================================="
echo "STEP 11 - Link Shared Library"
echo "=============================================="

riscv64-unknown-linux-gnu-gcc -shared \
  $BUILD_DIR/resnet20_riscv_rvv_taffo.o \
  $RUNTIME_OBJ_DIR/*.o \
  -o $BUILD_DIR/resnet20_riscv_rvv_taffo.so \
  -march=rv64gc -mabi=lp64d

echo "Step 11 done -> $BUILD_DIR/resnet20_riscv_rvv_taffo.so"
echo ""

# ============================================================
# Step 12 - Build Standalone Test Executable
# ============================================================

echo "=============================================="
echo "STEP 12 - Build Standalone Test Executable"
echo "=============================================="

if [ ! -f "$DRIVER_SRC" ]; then
  echo "  SKIPPED: no driver source found at $DRIVER_SRC"
  echo "  A driver (resnet20.cpp) with a real CIFAR-10 test image, matching"
  echo "  the [1,3,32,32] input shape, still needs to be written."
else
  riscv64-unknown-linux-gnu-g++ \
    --std=c++11 -O3 \
    $DRIVER_SRC \
    $BUILD_DIR/resnet20_riscv_rvv_taffo.o \
    $RUNTIME_OBJ_DIR/*.o \
    -o $BUILD_DIR/resnet20_riscv_rvv_taffo \
    -I $ONNX_MLIR_INCLUDE \
    -march=rv64gcv -mabi=lp64d -static

  echo "Step 12 done -> $BUILD_DIR/resnet20_riscv_rvv_taffo (statically linked)"
fi
echo ""

echo "=============================================="
echo "ALL STEPS COMPLETE"
echo "=============================================="
