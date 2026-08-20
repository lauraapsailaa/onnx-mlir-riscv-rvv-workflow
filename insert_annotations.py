#!/usr/bin/env python3
"""
Insert TAFFO annotation calls (`func.call @set_range(...)`) into a pre-raised
MLIR file, using exact min/max/precision bounds derived from the model's
actual weight/bias data (see extract_ranges.py) and a chosen input range.

--raise-to-taffo (TAFFO-MLIR's RaiseToTaffoPass.cpp, RewriteSetRangeCall)
recognizes any func.call whose callee name contains "set_range" and converts
it directly into a taffo.cast2real with the given min/max/precision operands.
This script inserts such calls right after every memref.load/affine.load that
reads from a matched krnl.global (weight/bias) or the model's input tensor.

Weight/bias tensors are matched to krnl.global ops BY SHAPE (krnl.global does
not preserve the original ONNX initializer name). If your model has two
different weight tensors with the same shape, this will not disambiguate
between them -- check the printed mapping before trusting the output.

IMPORTANT: MLIR text emitted by this pipeline reuses local SSA names (e.g.
%14) independently inside each nested loop body. This script tracks brace
depth so a renamed value is only substituted within the same lexical block
as its definition, never across sibling loop bodies that happen to reuse
the same local name.

Usage:
    python3 insert_annotations.py \
        mnist_pre_taffo_fixed.mlir \
        mnist_ranges.json \
        -o mnist_pre_taffo_annotated.mlir

Always inspect the output diff before feeding it into --raise-to-taffo.
"""

import argparse
import json
import re
import sys


GLOBAL_RE = re.compile(
    r'^\s*(%\S+)\s*=\s*"krnl\.global"\(\).*?shape\s*=\s*\[([\d,\s]*)\]'
)
LOAD_RE = re.compile(
    r'^(?P<indent>\s*)(?P<result>%\S+)\s*=\s*(?P<op>memref\.load|affine\.load)\s+'
    r'(?P<memref>%\S+)\[(?P<indices>[^\]]*)\]\s*:\s*memref<[^>]*>'
)
REINTERPRET_RE = re.compile(
    r'^\s*(%\S+)\s*=\s*memref\.reinterpret_cast\s+(%\S+)'
)
MAXNUMF_RE = re.compile(
    r'^(?P<indent>\s*)(?P<result>%\S+)\s*=\s*arith\.maxnumf\s+%\S+,\s*%\S+\s*:\s*f32'
)
MATH_EXP_RE = re.compile(
    r'^(?P<indent>\s*)(?P<result>%\S+)\s*=\s*math\.exp\s+%\S+\s*:\s*f32'
)
ACCUM_LOAD_RE = re.compile(
    r'^(?P<indent>\s*)(?P<result>%\S+)\s*=\s*affine\.load\s+(?P<memref>%\S+)\[\]\s*:\s*memref<f32>'
)


def parse_shape(shape_str):
    if not shape_str.strip():
        return ()
    return tuple(int(x.strip()) for x in shape_str.split(",") if x.strip())


def load_ranges(json_path):
    with open(json_path) as f:
        return json.load(f)


def find_krnl_globals(lines):
    """Map SSA result name -> declared shape tuple, for every krnl.global op."""
    result = {}
    for line in lines:
        m = GLOBAL_RE.match(line)
        if m:
            name, shape_str = m.group(1), m.group(2)
            result[name] = parse_shape(shape_str)
    return result


def find_image_aliases(lines, arg0_name="%arg0", extra_seeds=None):
    """
    Return the set of SSA names that alias the model's input tensor:
    %arg0 itself, plus anything derived from it (or from any extra seed
    name, e.g. a range-preserving intermediate buffer like a MaxPool
    output) via memref.reinterpret_cast, directly or transitively.
    """
    aliases = {arg0_name}
    if extra_seeds:
        aliases.update(extra_seeds)
    changed = True
    while changed:
        changed = False
        for line in lines:
            m = REINTERPRET_RE.match(line)
            if m:
                dst, src = m.group(1), m.group(2)
                if src in aliases and dst not in aliases:
                    aliases.add(dst)
                    changed = True
    return aliases


def build_shape_to_range(ranges, input_tensor_name):
    """
    Map shape tuple -> (min, max, precision, label) for every weight/bias
    entry in the ranges JSON (excludes the input tensor, handled separately
    via alias tracing).
    """
    result = {}
    for name, info in ranges.items():
        if name == input_tensor_name:
            continue
        shape = tuple(info.get("shape") or ())
        if shape in result:
            print(f"WARNING: shape {shape} matches multiple tensors "
                  f"({result[shape][3]!r} and {name!r}) -- disambiguation not possible, "
                  f"keeping first match", file=sys.stderr)
            continue
        result[shape] = (info["min"], info["max"], info["precision"], name)
    return result


def compute_brace_depths(lines):
    depths = []
    depth = 0
    for line in lines:
        depth += line.count("{") - line.count("}")
        depths.append(depth)
    return depths


def find_scope_end(depths, start_idx, base_depth):
    for i in range(start_idx + 1, len(depths)):
        if depths[i] < base_depth:
            return i
    return len(depths)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mlir_path")
    parser.add_argument("ranges_json")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--input-tensor-name", default="image",
                         help="Key in the ranges JSON for the model's input tensor (default: image)")
    parser.add_argument("--extra-input-alias", action="append", default=[], metavar="SSA_NAME",
                         help="Additional memref SSA names (e.g. '%%alloc') to treat as carrying "
                              "the same range as the input tensor. Use this for buffers that are "
                              "range-preserving copies of the input (e.g. a MaxPool output) but "
                              "aren't detected by automatic memref.reinterpret_cast alias tracing "
                              "because they're written to by a loop rather than derived directly "
                              "from %%arg0. Repeatable.")
    parser.add_argument("--op-range", action="append", default=[], metavar="LINE:MIN,MAX",
                         help="Override the range for a specific arith.maxnumf or math.exp op, "
                              "identified by its 1-indexed line number in the INPUT mlir file "
                              "(matching what 'grep -n' reports). Repeatable. "
                              "Example: --op-range 23:0.0,1.0 --op-range 57:0.0,59.22 "
                              "--op-range 84:-3439.75,2726.96 --op-range 90:0.0,1.0")
    parser.add_argument("--accumulator-range", default=None, metavar="MIN,MAX",
                         help="A single min,max bound applied to EVERY zero-dimension scalar "
                              "accumulator load (pattern: 'affine.load %%X[] : memref<f32>'), "
                              "e.g. the reduction accumulator inside a matmul loop. Use the "
                              "widest layer range you've computed (e.g. the final layer's "
                              "pre-activation sum) since it's a valid superset for every "
                              "narrower accumulator elsewhere in the network. "
                              "Example: --accumulator-range=-3439.75,2726.96")
    parser.add_argument("--buffer-range", action="append", default=[], metavar="SSA_NAME:MIN,MAX",
                         help="Annotate every load from a specific named intermediate memref "
                              "buffer with an explicit range. Use this for buffers that hold "
                              "an already-computed value (e.g. a layer's activation output "
                              "stored via memref.store) which is then read back by a LATER "
                              "op -- the original annotation on the value doesn't survive a "
                              "store/load round-trip through memory, since MLIR's dataflow "
                              "analysis tracks SSA values, not memory contents. Repeatable. "
                              "Example: --buffer-range '%%alloc_2:0.0,59.22'")
    args = parser.parse_args()

    with open(args.mlir_path) as f:
        lines = f.read().splitlines()

    ranges = load_ranges(args.ranges_json)
    shape_to_range = build_shape_to_range(ranges, args.input_tensor_name)

    krnl_globals = find_krnl_globals(lines)
    print(f"Found {len(krnl_globals)} krnl.global op(s):", file=sys.stderr)
    krnl_global_ranges = {}
    for ssa_name, shape in krnl_globals.items():
        match = shape_to_range.get(shape)
        if match:
            krnl_global_ranges[ssa_name] = match
            print(f"  {ssa_name} shape={shape} -> {match[3]} "
                  f"[{match[0]}, {match[1]}]", file=sys.stderr)
        else:
            print(f"  {ssa_name} shape={shape} -> NO MATCH in ranges JSON "
                  f"(will be left unannotated)", file=sys.stderr)

    image_aliases = find_image_aliases(lines, extra_seeds=args.extra_input_alias)
    print(f"Input-aliased memrefs: {image_aliases}", file=sys.stderr)
    input_info = ranges.get(args.input_tensor_name)
    if not input_info:
        print(f"WARNING: no entry '{args.input_tensor_name}' in ranges JSON; "
              f"input will be left unannotated", file=sys.stderr)

    op_range_overrides = {}  # 1-indexed line number -> (min, max, precision, label)
    for spec in args.op_range:
        try:
            line_str, bounds_str = spec.split(":", 1)
            min_str, max_str = bounds_str.split(",")
            line_no = int(line_str)
            op_range_overrides[line_no] = (float(min_str), float(max_str), 0.01,
                                            f"line{line_no}")
        except ValueError:
            print(f"ERROR: could not parse --op-range spec {spec!r}, expected LINE:MIN,MAX",
                  file=sys.stderr)
            sys.exit(1)
    if op_range_overrides:
        print(f"Per-line op-range overrides:", file=sys.stderr)
        for line_no, (vmin, vmax, _, _) in sorted(op_range_overrides.items()):
            print(f"  line {line_no}: [{vmin}, {vmax}]", file=sys.stderr)
    else:
        print("No --op-range overrides given; arith.maxnumf/math.exp outputs "
              "will be left unannotated", file=sys.stderr)

    accumulator_range = None
    if args.accumulator_range:
        amin_s, amax_s = args.accumulator_range.split(",")
        accumulator_range = (float(amin_s), float(amax_s), 0.01, "accumulator")
        print(f"Accumulator range (applied to all scalar accumulators): "
              f"[{accumulator_range[0]}, {accumulator_range[1]}]", file=sys.stderr)
    else:
        print("No --accumulator-range given; scalar accumulator loads "
              "will be left unannotated", file=sys.stderr)

    buffer_ranges = {}  # SSA name -> (min, max, precision, label)
    for spec in args.buffer_range:
        try:
            name, bounds_str = spec.split(":", 1)
            min_str, max_str = bounds_str.split(",")
            buffer_ranges[name] = (float(min_str), float(max_str), 0.01, f"buffer{name}")
        except ValueError:
            print(f"ERROR: could not parse --buffer-range spec {spec!r}, "
                  f"expected SSA_NAME:MIN,MAX", file=sys.stderr)
            sys.exit(1)
    if buffer_ranges:
        print(f"Named buffer-range overrides:", file=sys.stderr)
        for name, (vmin, vmax, _, _) in buffer_ranges.items():
            print(f"  {name}: [{vmin}, {vmax}]", file=sys.stderr)

    depths = compute_brace_depths(lines)

    out_lines = []
    counter = 0
    matched_count = 0
    accumulator_count = 0
    active_renames = []  # list of (scope_end_idx, old_name, new_name)

    def emit_annotation(indent, result_name, range_info, base_depth, line_idx):
        nonlocal counter
        vmin, vmax, prec, label = range_info
        counter += 1
        min_v = f"%rmin_{counter}"
        max_v = f"%rmax_{counter}"
        prec_v = f"%rprec_{counter}"
        annotated_v = f"%rannot_{counter}"
        out_lines.append(f'{indent}{min_v} = arith.constant {vmin!r} : f64')
        out_lines.append(f'{indent}{max_v} = arith.constant {vmax!r} : f64')
        out_lines.append(f'{indent}{prec_v} = arith.constant {prec!r} : f64')
        out_lines.append(
            f'{indent}{annotated_v} = func.call @set_range({result_name}, '
            f'{min_v}, {max_v}, {prec_v}) : (f32, f64, f64, f64) -> f32  '
            f'// annotate {label}'
        )
        scope_end = find_scope_end(depths, line_idx, base_depth)
        active_renames.append((scope_end, result_name, annotated_v))

    for i, orig_line in enumerate(lines):
        line = orig_line
        active_renames = [r for r in active_renames if r[0] > i]
        for scope_end, old_name, new_name in active_renames:
            line = re.sub(re.escape(old_name) + r'\b', new_name, line)

        out_lines.append(line)

        ma = ACCUM_LOAD_RE.match(orig_line)
        if ma:
            if accumulator_range:
                accumulator_count += 1
                emit_annotation(ma.group("indent"), ma.group("result"),
                                 accumulator_range, depths[i], i)
            continue

        m = LOAD_RE.match(orig_line)
        if m:
            memref_name = m.group("memref")
            result_name = m.group("result")
            base_depth = depths[i]

            range_info = None
            if memref_name in buffer_ranges:
                range_info = buffer_ranges[memref_name]
            elif memref_name in image_aliases and input_info:
                range_info = (input_info["min"], input_info["max"],
                              input_info["precision"], args.input_tensor_name)
            elif memref_name in krnl_global_ranges:
                range_info = krnl_global_ranges[memref_name]

            if range_info:
                emit_annotation(m.group("indent"), result_name, range_info, base_depth, i)
            continue

        mm = MAXNUMF_RE.match(orig_line)
        me = MATH_EXP_RE.match(orig_line)
        matched = mm or me
        if matched:
            line_no = i + 1  # 1-indexed, matching grep -n convention
            override = op_range_overrides.get(line_no)
            if override:
                matched_count += 1
                emit_annotation(matched.group("indent"), matched.group("result"),
                                 override, depths[i], i)
            else:
                op_kind = "arith.maxnumf" if mm else "math.exp"
                print(f"  NOTE: line {line_no} ({op_kind}) has no --op-range override, "
                      f"leaving unannotated", file=sys.stderr)

    decl = 'func.func private @set_range(f32, f64, f64, f64) -> f32'
    inserted = False
    for idx, l in enumerate(out_lines):
        if l.strip().startswith("module"):
            out_lines.insert(idx + 1, "  " + decl)
            inserted = True
            break
    if not inserted:
        out_lines.insert(0, decl)

    with open(args.output, "w") as f:
        f.write("\n".join(out_lines) + "\n")

    print(f"\nWrote {args.output} ({counter} annotation(s) inserted: "
          f"{matched_count} maxnumf/exp via --op-range, "
          f"{accumulator_count} accumulator(s))", file=sys.stderr)


if __name__ == "__main__":
    main()
