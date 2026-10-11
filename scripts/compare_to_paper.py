"""Compare regenerated results with the canonical paper results in paper_assets/.

Usage (from the repository root, after the paper run)::

    python scripts/compare_to_paper.py
    python scripts/compare_to_paper.py --outputs outputs --paper paper_assets --tol 1e-9

For every compared CSV the rows are aligned on their key columns and each
column is checked:

* deterministic numeric columns must agree to ``--tol`` (absolute, default
  1e-9); NaN must line up with NaN;
* text/boolean columns must be identical;
* timing and memory columns (``time_s*``, ``peak_mem_kib*``) depend on the
  machine and are NOT part of PASS/FAIL -- their differences are reported in a
  separate table.

Columns that exist only in the regenerated file (e.g. the Holm-adjusted
``p_holm`` added after the paper run) are listed as "new" and not compared.

Exit status is 0 when every compared column passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# file name -> key columns used to align rows
FILES = {
    "results_aggregate.csv": ["scenario", "algorithm"],
    "wilcoxon_J.csv": ["scenario", "metric", "alg_a", "alg_b"],
    "results_per_run.csv": ["scenario", "algorithm", "run"],
}
# Required by the paper check; results_per_run.csv is compared when present.
REQUIRED = {"results_aggregate.csv", "wilcoxon_J.csv"}
TIMING_PREFIXES = ("time_s", "peak_mem_kib")


def _is_timing(col: str) -> bool:
    return col.startswith(TIMING_PREFIXES)


def _compare_file(name: str, keys: list[str], new_path: Path, ref_path: Path, tol: float):
    """Return (rows, timing_rows, ok) for one CSV pair."""
    new = pd.read_csv(new_path)
    ref = pd.read_csv(ref_path)
    rows, timing_rows = [], []

    # --- row alignment -------------------------------------------------
    new_keys = set(map(tuple, new[keys].astype(str).to_numpy()))
    ref_keys = set(map(tuple, ref[keys].astype(str).to_numpy()))
    if new_keys != ref_keys:
        rows.append(
            (
                name,
                "<row keys>",
                "-",
                "FAIL",
                f"missing={sorted(ref_keys - new_keys)} extra={sorted(new_keys - ref_keys)}",
            )
        )
        return rows, timing_rows, False
    new = new.set_index(keys).sort_index()
    ref = ref.set_index(keys).sort_index()
    new = new.loc[ref.index]

    ok = True
    for col in ref.columns:
        if col not in new.columns:
            rows.append((name, col, "-", "FAIL", "column missing in outputs"))
            ok = False
            continue
        a, b = new[col], ref[col]
        numeric = (
            pd.api.types.is_numeric_dtype(a)
            and pd.api.types.is_numeric_dtype(b)
            and not pd.api.types.is_bool_dtype(b)
        )
        if numeric:
            av, bv = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
            nan_ok = np.array_equal(np.isnan(av), np.isnan(bv))
            both = ~np.isnan(av) & ~np.isnan(bv)
            diff = float(np.max(np.abs(av[both] - bv[both]))) if both.any() else 0.0
            if _is_timing(col):
                rel = (
                    (np.abs(av[both] - bv[both]) / np.maximum(np.abs(bv[both]), 1e-12))
                    if both.any()
                    else []
                )
                timing_rows.append(
                    (name, col, f"{diff:.4g}", f"{float(np.max(rel)) if len(rel) else 0.0:.1%}")
                )
                continue
            passed = nan_ok and diff <= tol
            detail = "" if nan_ok else "NaN pattern differs"
            rows.append((name, col, f"{diff:.3g}", "PASS" if passed else "FAIL", detail))
        else:
            same = (a.fillna("").astype(str) == b.fillna("").astype(str)).all()
            passed = bool(same)
            rows.append((name, col, "exact", "PASS" if passed else "FAIL", ""))
        ok &= passed
    for col in new.columns:
        if col not in ref.columns:
            rows.append((name, col, "-", "NEW", "not in paper_assets; not compared"))
    return rows, timing_rows, ok


def _print_table(header, rows):
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*header))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print(fmt.format(*r))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outputs", default="outputs", help="regenerated results dir")
    ap.add_argument("--paper", default="paper_assets", help="canonical results dir")
    ap.add_argument(
        "--tol", type=float, default=1e-9, help="absolute tolerance for deterministic metrics"
    )
    args = ap.parse_args(argv)
    out_dir, ref_dir = Path(args.outputs), Path(args.paper)

    all_rows, all_timing, overall = [], [], True
    for name, keys in FILES.items():
        new_p, ref_p = out_dir / name, ref_dir / name
        if not new_p.exists() or not ref_p.exists():
            if name in REQUIRED:
                all_rows.append(
                    (
                        name,
                        "<file>",
                        "-",
                        "FAIL",
                        f"missing: {new_p if not new_p.exists() else ref_p}",
                    )
                )
                overall = False
            continue
        rows, timing, ok = _compare_file(name, keys, new_p, ref_p, args.tol)
        all_rows += rows
        all_timing += timing
        overall &= ok

    print(
        f"Comparing {out_dir}/ against {ref_dir}/  (tolerance {args.tol:g}, "
        "timing/memory excluded)\n"
    )
    _print_table(("file", "column", "max |diff|", "result", "note"), all_rows)
    if all_timing:
        print("\nTiming / memory (machine-dependent, informational only):\n")
        _print_table(("file", "column", "max |diff|", "max rel diff"), all_timing)
    n_fail = sum(r[3] == "FAIL" for r in all_rows)
    n_pass = sum(r[3] == "PASS" for r in all_rows)
    print(f"\nOVERALL: {'PASS' if overall else 'FAIL'}  ({n_pass} columns passed, {n_fail} failed)")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
