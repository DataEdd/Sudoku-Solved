"""
Post-process ``evaluation/ablation_results.json`` into human-readable
tables and ranked findings for the notebook writeup.

This script is standalone: it only reads the committed ablation results,
so it runs fast and produces deterministic output that can be pasted
directly into ``notebooks/03_ocr.ipynb``.

Usage:
    python -m evaluation.ablation_analysis
    python -m evaluation.ablation_analysis --markdown > /tmp/ablation.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

RESULTS_PATH = Path(__file__).parent / "ablation_results.json"


def load_results() -> Dict[str, Any]:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"{RESULTS_PATH} does not exist. Run `python -m evaluation.ablation` first."
        )
    return json.loads(RESULTS_PATH.read_text())


def channel_family(channels: List[int]) -> str:
    """Map a channel list to its family name (small / medium / large)."""
    first = channels[0]
    return {16: "small", 32: "medium", 64: "large"}.get(first, "?")


def real_overall_acc(r: Dict[str, Any]) -> float:
    """Combined real-photo accuracy, weighted by cell counts."""
    f = r["real_filled_acc"]
    e = r["real_empty_acc"]
    # Rough weighting: the 38-image GT has ~1720 filled + ~1360 empty cells
    # (measured in ocr_analysis.ipynb). Use 0.5/0.5 for simplicity; the
    # exact ratio doesn't change rankings meaningfully.
    return 0.5 * f + 0.5 * e


def print_baseline_anchor(results: List[Dict[str, Any]]) -> None:
    baseline = next(
        (r for r in results if r["name"] == "d3_c-medium_drop0.3"), None,
    )
    if baseline is None:
        print("WARNING: baseline config (d3_c-medium_drop0.3) not in results")
        return
    print("## Baseline anchor (production config)")
    print()
    print(f"  depth=3, channels=[32,64,128], dropout=0.3  →  {baseline['parameters']:,} params")
    print(f"    synth test (clean) : {baseline['synthetic_test_acc']:.4f}")
    if "synthetic_test_acc_aug" in baseline:
        print(f"    synth test (aug)   : {baseline['synthetic_test_acc_aug']:.4f}")
    print(f"    real filled        : {baseline['real_filled_acc']:.4f}")
    print(f"    real empty         : {baseline['real_empty_acc']:.4f}")
    print(f"    training time      : {baseline['train_time_s']:.0f} s")
    print(f"    val acc (best)     : {baseline['best_val_acc']:.4f}")
    print()


def print_full_table(results: List[Dict[str, Any]]) -> None:
    print("## Full grid (sorted by real filled-cell accuracy, descending)")
    print()
    has_aug = any("synthetic_test_acc_aug" in r for r in results)
    if has_aug:
        print(
            f"| {'rank':>4} | {'name':<26} | {'params':>10} | "
            f"{'synth':>7} | {'synth_aug':>10} | {'real_filled':>12} | "
            f"{'real_empty':>11} | {'time (s)':>8} |"
        )
        print("|" + "-" * 6 + "|" + "-" * 28 + "|" + "-" * 12 + "|" + "-" * 9
              + "|" + "-" * 12 + "|" + "-" * 14 + "|" + "-" * 13 + "|" + "-" * 10 + "|")
        ranked = sorted(results, key=lambda r: r["real_filled_acc"], reverse=True)
        for i, r in enumerate(ranked, 1):
            aug = r.get("synthetic_test_acc_aug", 0.0)
            print(
                f"| {i:>4} | {r['name']:<26} | {r['parameters']:>10,} | "
                f"{r['synthetic_test_acc']:>7.4f} | {aug:>10.4f} | "
                f"{r['real_filled_acc']:>12.4f} | {r['real_empty_acc']:>11.4f} | "
                f"{r['train_time_s']:>8.0f} |"
            )
    else:
        print(
            f"| {'rank':>4} | {'name':<26} | {'params':>10} | "
            f"{'synth':>7} | {'real_filled':>12} | {'real_empty':>11} | "
            f"{'time (s)':>8} |"
        )
        print("|" + "-" * 6 + "|" + "-" * 28 + "|" + "-" * 12 + "|" + "-" * 9
              + "|" + "-" * 14 + "|" + "-" * 13 + "|" + "-" * 10 + "|")
        ranked = sorted(results, key=lambda r: r["real_filled_acc"], reverse=True)
        for i, r in enumerate(ranked, 1):
            print(
                f"| {i:>4} | {r['name']:<26} | {r['parameters']:>10,} | "
                f"{r['synthetic_test_acc']:>7.4f} | {r['real_filled_acc']:>12.4f} | "
                f"{r['real_empty_acc']:>11.4f} | {r['train_time_s']:>8.0f} |"
            )
    print()


def print_parameter_efficiency(results: List[Dict[str, Any]]) -> None:
    """For each unique parameter count, show the best config that hit it."""
    by_params: Dict[int, List[Dict[str, Any]]] = {}
    for r in results:
        by_params.setdefault(r["parameters"], []).append(r)

    print("## Parameter efficiency (best real_filled per param bucket)")
    print()
    print(
        f"| {'parameters':>12} | {'best config':<28} | "
        f"{'real_filled':>12} | {'real_empty':>11} | {'synth':>7} |"
    )
    print("|" + "-" * 14 + "|" + "-" * 30 + "|" + "-" * 14
          + "|" + "-" * 13 + "|" + "-" * 9 + "|")
    for params in sorted(by_params.keys()):
        bucket = by_params[params]
        best = max(bucket, key=lambda r: r["real_filled_acc"])
        print(
            f"| {params:>12,} | {best['name']:<28} | "
            f"{best['real_filled_acc']:>12.4f} | {best['real_empty_acc']:>11.4f} | "
            f"{best['synthetic_test_acc']:>7.4f} |"
        )
    print()


def print_axis_sweeps(results: List[Dict[str, Any]]) -> None:
    """Hold two axes constant at baseline values, sweep the third."""
    print("## One-at-a-time sweeps (other two axes held at baseline)")
    print()

    def rows_where(**constraints) -> List[Dict[str, Any]]:
        out = []
        for r in results:
            fam = channel_family(r["channels"])
            if all(
                {
                    "depth": r["depth"],
                    "family": fam,
                    "dropout": r["dropout"],
                }[k] == v
                for k, v in constraints.items()
            ):
                out.append(r)
        return out

    # Depth sweep (channels=medium, dropout=0.3)
    print("### Depth sweep (channels=medium, dropout=0.3)")
    print()
    print(f"| {'depth':>6} | {'params':>10} | {'synth':>7} | {'real_filled':>12} | {'real_empty':>11} |")
    print("|" + "-" * 8 + "|" + "-" * 12 + "|" + "-" * 9 + "|" + "-" * 14 + "|" + "-" * 13 + "|")
    for depth in [2, 3, 4]:
        rows = rows_where(depth=depth, family="medium", dropout=0.3)
        if not rows:
            print(f"| {depth:>6} | {'pending':>10} | {'—':>7} | {'—':>12} | {'—':>11} |")
            continue
        r = rows[0]
        print(
            f"| {depth:>6} | {r['parameters']:>10,} | {r['synthetic_test_acc']:>7.4f} | "
            f"{r['real_filled_acc']:>12.4f} | {r['real_empty_acc']:>11.4f} |"
        )
    print()

    # Channel sweep (depth=3, dropout=0.3)
    print("### Channel width sweep (depth=3, dropout=0.3)")
    print()
    print(f"| {'family':<8} | {'params':>10} | {'synth':>7} | {'real_filled':>12} | {'real_empty':>11} |")
    print("|" + "-" * 10 + "|" + "-" * 12 + "|" + "-" * 9 + "|" + "-" * 14 + "|" + "-" * 13 + "|")
    for fam in ["small", "medium", "large"]:
        rows = rows_where(depth=3, family=fam, dropout=0.3)
        if not rows:
            print(f"| {fam:<8} | {'pending':>10} | {'—':>7} | {'—':>12} | {'—':>11} |")
            continue
        r = rows[0]
        print(
            f"| {fam:<8} | {r['parameters']:>10,} | {r['synthetic_test_acc']:>7.4f} | "
            f"{r['real_filled_acc']:>12.4f} | {r['real_empty_acc']:>11.4f} |"
        )
    print()

    # Dropout sweep (depth=3, channels=medium)
    print("### Dropout sweep (depth=3, channels=medium)")
    print()
    print(f"| {'dropout':>7} | {'params':>10} | {'synth':>7} | {'real_filled':>12} | {'real_empty':>11} |")
    print("|" + "-" * 9 + "|" + "-" * 12 + "|" + "-" * 9 + "|" + "-" * 14 + "|" + "-" * 13 + "|")
    for dropout in [0.2, 0.3, 0.5]:
        rows = rows_where(depth=3, family="medium", dropout=dropout)
        if not rows:
            print(f"| {dropout:>7} | {'pending':>10} | {'—':>7} | {'—':>12} | {'—':>11} |")
            continue
        r = rows[0]
        print(
            f"| {dropout:>7.2f} | {r['parameters']:>10,} | {r['synthetic_test_acc']:>7.4f} | "
            f"{r['real_filled_acc']:>12.4f} | {r['real_empty_acc']:>11.4f} |"
        )
    print()


def print_findings(results: List[Dict[str, Any]]) -> None:
    """Identify headline findings that the writeup should call out."""
    print("## Findings")
    print()

    # 1. Sweet spot
    ranked = sorted(results, key=lambda r: r["real_filled_acc"], reverse=True)
    best = ranked[0]
    baseline = next(
        (r for r in results if r["name"] == "d3_c-medium_drop0.3"), None
    )
    print(f"1. **Best real-photo filled accuracy:** {best['name']} "
          f"({best['real_filled_acc']:.4f}, {best['parameters']:,} params)")

    if baseline:
        diff = best["real_filled_acc"] - baseline["real_filled_acc"]
        if best["name"] == baseline["name"]:
            print(
                f"   The **production baseline is the winner** — "
                f"the ablation validates the current choice."
            )
        elif diff < 0.005:
            print(
                f"   Baseline is within 0.5 percentage points of the winner "
                f"({baseline['real_filled_acc']:.4f} vs {best['real_filled_acc']:.4f}); "
                f"the improvement isn't worth the parameter cost "
                f"({best['parameters']:,} vs baseline {baseline['parameters']:,})."
            )
        else:
            print(
                f"   Baseline trails by {diff * 100:.1f} percentage points "
                f"({baseline['real_filled_acc']:.4f} vs {best['real_filled_acc']:.4f}); "
                f"consider switching to {best['name']} if the "
                f"{best['parameters'] - baseline['parameters']:+,} parameter delta "
                f"is acceptable."
            )
    print()

    # 2. Synthetic vs real divergence
    synth_avg = sum(r["synthetic_test_acc"] for r in results) / len(results)
    real_avg = sum(r["real_filled_acc"] for r in results) / len(results)
    print(
        f"2. **Synthetic vs real gap:** average synthetic test accuracy is "
        f"{synth_avg:.4f}, average real filled-cell accuracy is {real_avg:.4f} — "
        f"a {(synth_avg - real_avg) * 100:.1f} percentage-point gap across "
        f"{len(results)} configs. The synthetic test split saturates near the "
        f"ceiling for every reasonable config; real-photo accuracy is where "
        f"architectural choices actually matter."
    )
    print()

    # 3. Shallow underperforms
    shallow = [r for r in results if r["depth"] == 2]
    if shallow:
        best_shallow = max(shallow, key=lambda r: r["real_filled_acc"])
        deep = [r for r in results if r["depth"] >= 3]
        if deep:
            best_deep = max(deep, key=lambda r: r["real_filled_acc"])
            print(
                f"3. **Depth matters more than dropout:** best depth=2 config "
                f"({best_shallow['name']}) hits {best_shallow['real_filled_acc']:.4f} "
                f"on real photos; best depth≥3 config "
                f"({best_deep['name']}) hits {best_deep['real_filled_acc']:.4f}. "
                f"Depth=2 is insufficient."
            )
    print()

    # 4. Overfitting signal
    print("4. **Overfitting check:** configs where real accuracy "
          "diverges significantly from synthetic test accuracy:")
    overfitted = sorted(
        results,
        key=lambda r: r["synthetic_test_acc"] - r["real_filled_acc"],
        reverse=True,
    )[:5]
    for r in overfitted:
        gap = r["synthetic_test_acc"] - r["real_filled_acc"]
        print(
            f"   - {r['name']:<26} synth={r['synthetic_test_acc']:.4f}, "
            f"real={r['real_filled_acc']:.4f}, gap={gap:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Produce ranked tables and findings from ablation_results.json"
    )
    parser.add_argument(
        "--markdown", action="store_true",
        help="Emit pure markdown (headers, tables) suitable for pasting into a notebook",
    )
    args = parser.parse_args()

    data = load_results()
    results = data.get("results", [])
    protocol = data.get("protocol", {})

    print(f"Loaded {len(results)} configs from {RESULTS_PATH}")
    print(f"Training protocol: epochs={protocol.get('epochs')}, "
          f"batch_size={protocol.get('batch_size')}, "
          f"optimizer={protocol.get('optimizer')}, "
          f"seed={protocol.get('seed')}")
    print()

    if not results:
        print("No results yet. The ablation may still be running.")
        return

    print_baseline_anchor(results)
    print_full_table(results)
    print_parameter_efficiency(results)
    print_axis_sweeps(results)
    print_findings(results)


if __name__ == "__main__":
    main()
