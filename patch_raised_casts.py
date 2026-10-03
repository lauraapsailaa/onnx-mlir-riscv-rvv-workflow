#!/usr/bin/env python3
"""
Post-raising annotation patch: raise-to-taffo generates a fresh, unbounded
`builtin.unrealized_conversion_cast %X : f32 to !taffo.real` for the RESULT
of certain loops (e.g. a max-reduction or sum-reduction affine.for/scf.for
whose body computes a properly-ranged value, but whose overall loop result,
used just after the loop closes, gets re-cast from scratch with no
annotation). This cast has no counterpart in the pre-raised MLIR -- raising
invents it -- so it can't be annotated by insert_annotations.py, which only
edits the pre-raised file.

This script operates directly on the AFTER-RAISING MLIR. Rather than relying
on manually-hunted line numbers (fragile: legitimate, already-safe in-loop
casts of ordinary values are textually interleaved with the problematic
ones), it AUTOMATICALLY detects the problematic pattern structurally: a
`builtin.unrealized_conversion_cast %Y : f32 to !taffo.real` where %Y is
itself the direct result of an `affine.for`/`scf.for` loop (as opposed to a
block argument or an ordinary arith result, which are safe and shouldn't be
touched). Each detected occurrence is reported with the declaring loop's own
line number, which you then supply a range for via --loop-range.

Usage:
    # First, discover what needs a range:
    python3 patch_raised_casts.py mnist_after_raise.mlir --list

    # Then patch, keyed by the LOOP's declaration line number:
    python3 patch_raised_casts.py mnist_after_raise.mlir \
        --loop-range 151:-3439.75,2726.96 \
        --loop-range 169:1.0,10.0 \
        -o mnist_after_raise_patched.mlir
"""

import argparse
import re
import sys


CAST_RE = re.compile(
    r'^(?P<indent>\s*)(?P<result>%\S+)\s*=\s*builtin\.unrealized_conversion_cast\s+'
    r'(?P<operand>%\S+)\s*:\s*f32\s+to\s+!taffo\.real\s*$'
)
LOOP_DECL_RE = re.compile(
    r'^(?P<indent>\s*)(?P<result>%\S+)\s*=\s*(?:affine\.for|scf\.for)\s+.*'
    r'iter_args\((?P<itervar>%\S+)\s*=\s*%\S+\).*->\s*\(f32\)'
)


def find_loop_results(lines):
    """Return a list of (result_name, declaring_loop_line, body_end_idx) for
    every affine.for/scf.for that produces a single f32 result. Tracks each
    loop's own body-end index (via brace depth) rather than a simple
    name->line dict, since result names like %106 are commonly reused
    across many structurally-unrelated loops (e.g. every conv layer's own
    reduction loop) -- a name-only mapping would let a later declaration
    silently overwrite an earlier one, misattributing every matching cast
    in the file to whichever loop happened to be declared last."""
    depths = []
    depth = 0
    for line in lines:
        depth += line.count("{") - line.count("}")
        depths.append(depth)

    result = []  # list of (result_name, loop_line, body_end_idx)
    for i, line in enumerate(lines):
        m = LOOP_DECL_RE.match(line)
        if m:
            base_depth = depths[i]
            body_end = len(lines)
            for j in range(i + 1, len(lines)):
                if depths[j] < base_depth:
                    body_end = j
                    break
            result.append((m.group("result"), i + 1, body_end))
    return result


def find_loop_iterargs(lines):
    """Map iter_arg block-argument name -> (declaring loop line, body start
    index, body end index) for every affine.for/scf.for with a single f32
    iter_arg, so casts of that specific block argument INSIDE the loop body
    can be found without confusing it with same-named iter_args in sibling
    loops (block-argument names like %arg3 are commonly reused)."""
    depths = []
    depth = 0
    for line in lines:
        depth += line.count("{") - line.count("}")
        depths.append(depth)

    result = []  # list of (itervar_name, loop_line, body_start_idx, body_end_idx)
    for i, line in enumerate(lines):
        m = LOOP_DECL_RE.match(line)
        if m:
            base_depth = depths[i]
            body_end = len(lines)
            for j in range(i + 1, len(lines)):
                if depths[j] < base_depth:
                    body_end = j
                    break
            result.append((m.group("itervar"), i + 1, i, body_end))
    return result


def find_problematic_casts(lines, loop_results):
    """
    Return a list of (cast_line_no, declaring_loop_line_no, cast_match)
    for every unrealized_conversion_cast whose operand is a loop's own
    result -- the structurally problematic pattern. For each cast, matches
    against the loop (among possibly several sharing the same result name
    in unrelated scopes) whose body ends nearest to, and at or before, the
    cast's own line -- since a loop's result is only ever used starting
    immediately after its own closing brace, this correctly disambiguates
    same-named results from different loops.
    """
    found = []
    for i, line in enumerate(lines):
        m = CAST_RE.match(line)
        if not m:
            continue
        operand = m.group("operand")
        best = None
        for result_name, loop_line, body_end in loop_results:
            if result_name != operand:
                continue
            if body_end > i + 1:  # loop must close at or before this cast
                continue
            if best is None or body_end > best[1]:
                best = (loop_line, body_end)
        if best is not None:
            found.append((i + 1, best[0], m))
    return found


def find_problematic_entry_casts(lines, loop_iterargs):
    """
    Return a list of (cast_line_no, declaring_loop_line_no, cast_match) for
    every unrealized_conversion_cast whose operand is a loop's OWN iter_arg
    block argument, found strictly within that loop's own body (so a
    same-named iter_arg in an unrelated sibling loop is never confused for
    this one -- block-argument names like %arg3 are commonly reused).
    """
    found = []
    for itervar, loop_line, body_start, body_end in loop_iterargs:
        for i in range(body_start, body_end):
            m = CAST_RE.match(lines[i])
            if m and m.group("operand") == itervar:
                found.append((i + 1, loop_line, m))
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mlir_path")
    parser.add_argument("-o", "--output",
                         help="Required unless --list is given")
    parser.add_argument("--list", action="store_true",
                         help="Just report detected problematic casts and their declaring "
                              "loop's line number, without patching anything.")
    parser.add_argument("--loop-range", action="append", default=[], metavar="LOOP_LINE:MIN,MAX",
                         help="Provide the range for the loop-RESULT cast belonging to the "
                              "loop declared at LOOP_LINE (1-indexed, as reported by --list "
                              "or by this script's own diagnostic output). Repeatable. "
                              "Example: --loop-range 151:-3439.75,2726.96")
    parser.add_argument("--iterarg-range", action="append", default=[], metavar="LOOP_LINE:MIN,MAX",
                         help="Provide the range for the loop-ENTRY cast of that loop's OWN "
                              "iter_arg block argument, found inside the loop body (a distinct "
                              "pattern from --loop-range: this is the accumulator's valid range "
                              "at ANY point during iteration, so its lower bound is usually "
                              "wider than the final result's -- e.g. an accumulator starting "
                              "at 0.0 needs [0.0, MAX], not the result's tighter [MIN, MAX]). "
                              "Repeatable. Example: --iterarg-range 167:0.0,10.0")
    parser.add_argument("--precision", type=float, default=0.01,
                         help="Precision/error term to use for all patched casts (default: 0.01)")
    args = parser.parse_args()

    if not args.list and not args.output:
        print("ERROR: -o/--output is required unless --list is given", file=sys.stderr)
        sys.exit(1)

    with open(args.mlir_path) as f:
        lines = f.read().splitlines()

    loop_results = find_loop_results(lines)
    loop_iterargs = find_loop_iterargs(lines)
    problematic = find_problematic_casts(lines, loop_results)
    problematic_entries = find_problematic_entry_casts(lines, loop_iterargs)

    print(f"Found {len(loop_results)} single-f32-result affine.for/scf.for loop(s).",
          file=sys.stderr)
    print(f"Found {len(problematic)} problematic loop-RESULT cast(s):", file=sys.stderr)
    for cast_line, loop_line, m in problematic:
        print(f"  cast at line {cast_line} (operand {m.group('operand')!r}, "
              f"result {m.group('result')!r}) -- declaring loop at line {loop_line}",
              file=sys.stderr)
    print(f"Found {len(problematic_entries)} problematic loop-ENTRY (iter_arg) cast(s):",
          file=sys.stderr)
    for cast_line, loop_line, m in problematic_entries:
        print(f"  cast at line {cast_line} (operand {m.group('operand')!r}, "
              f"result {m.group('result')!r}) -- declaring loop at line {loop_line}",
              file=sys.stderr)

    if args.list:
        if problematic or problematic_entries:
            print("\nSupply a range for each via --loop-range / --iterarg-range "
                  "LOOP_LINE:MIN,MAX (using the 'declaring loop' line number above).",
                  file=sys.stderr)
        return

    def parse_specs(specs, flag_name):
        overrides = {}
        for spec in specs:
            try:
                line_str, bounds_str = spec.split(":", 1)
                min_str, max_str = bounds_str.split(",")
                overrides[int(line_str)] = (float(min_str), float(max_str))
            except ValueError:
                print(f"ERROR: could not parse {flag_name} spec {spec!r}, "
                      f"expected LOOP_LINE:MIN,MAX", file=sys.stderr)
                sys.exit(1)
        return overrides

    result_overrides = parse_specs(args.loop_range, "--loop-range")
    entry_overrides = parse_specs(args.iterarg_range, "--iterarg-range")

    unmatched_results = set(l for _, l, _ in problematic) - set(result_overrides.keys())
    unmatched_entries = set(l for _, l, _ in problematic_entries) - set(entry_overrides.keys())
    if unmatched_results:
        print(f"\nWARNING: no --loop-range given for loop(s) declared at line(s) "
              f"{sorted(unmatched_results)} -- their result cast(s) will be left "
              f"unpatched.", file=sys.stderr)
    if unmatched_entries:
        print(f"\nWARNING: no --iterarg-range given for loop(s) declared at line(s) "
              f"{sorted(unmatched_entries)} -- their entry cast(s) will be left "
              f"unpatched.", file=sys.stderr)

    out_lines = list(lines)
    patched = 0
    for cast_line, loop_line, m in problematic:
        if loop_line not in result_overrides:
            continue
        vmin, vmax = result_overrides[loop_line]
        indent = m.group("indent")
        result = m.group("result")
        operand = m.group("operand")
        new_line = (f'{indent}{result} = taffo.cast2real {operand}, '
                    f'{args.precision!r}, {vmin!r}, {vmax!r} : f32 -> !taffo.real')
        out_lines[cast_line - 1] = new_line
        patched += 1
        print(f"  Patched RESULT cast line {cast_line}: {lines[cast_line - 1].strip()}\n"
              f"    -> {new_line.strip()}", file=sys.stderr)

    for cast_line, loop_line, m in problematic_entries:
        if loop_line not in entry_overrides:
            continue
        vmin, vmax = entry_overrides[loop_line]
        indent = m.group("indent")
        result = m.group("result")
        operand = m.group("operand")
        new_line = (f'{indent}{result} = taffo.cast2real {operand}, '
                    f'{args.precision!r}, {vmin!r}, {vmax!r} : f32 -> !taffo.real')
        out_lines[cast_line - 1] = new_line
        patched += 1
        print(f"  Patched ENTRY cast line {cast_line}: {lines[cast_line - 1].strip()}\n"
              f"    -> {new_line.strip()}", file=sys.stderr)

    with open(args.output, "w") as f:
        f.write("\n".join(out_lines) + "\n")

    total_found = len(problematic) + len(problematic_entries)
    print(f"\nWrote {args.output} ({patched}/{total_found} detected patches applied)",
          file=sys.stderr)


if __name__ == "__main__":
    main()