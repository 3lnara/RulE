"""
summarize_ablation.py — Parse ablation training logs and print a metric table.

Reads training.log files produced by train_beta_grounding.py from each variant
subdirectory and extracts MRR / Hit@1 / Hit@10.  Writes SUMMARY.txt to the
ablation root and prints the table to stdout.

Usage:
    python summarize_ablation.py --root ~/RulE/outputs/ablation_umls_seed42
    python summarize_ablation.py --root ~/RulE/outputs/ablation_umls_seed42 --no_write
"""

import argparse
import os
import re
import sys
from pathlib import Path


VARIANTS = [
    # (directory_name, results_block_header, display_label)
    ("fixed_alpha",          "Fixed Alpha",           "fixed_alpha"),
    ("perrel_only",          "Adaptive Beta (trained)", "perrel_only"),
    ("density_only",         "Adaptive Beta (trained)", "density_only"),
    ("maxscore_only",        "Adaptive Beta (trained)", "maxscore_only"),
    ("perrel_density_joint", "Adaptive Beta (trained)", "perrel_density_joint"),
    ("perrel_maxscore_joint","Adaptive Beta (trained)", "perrel_maxscore_joint"),
]

METRICS = ["MRR", "Hit@1", "Hit@10"]


def extract_block(text: str, header: str):
    """Return the text starting at the line containing `header`, up to 30 lines."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if header in line:
            return "\n".join(lines[i: i + 30])
    return None


def extract_metric(block: str, metric: str) -> str:
    """Pull the first float after `metric:` in `block`."""
    for line in block.splitlines():
        if metric in line and ":" in line:
            m = re.search(r"([0-9]+\.[0-9]+)", line)
            if m:
                return m.group(1)
    return "N/A"


def parse_log(log_path: Path, block_header: str) -> dict[str, str]:
    if not log_path.exists():
        return {m: "MISSING" for m in METRICS}
    text = log_path.read_text(errors="replace")
    block = extract_block(text, block_header)
    if block is None:
        return {m: "NO_BLOCK" for m in METRICS}
    return {m: extract_metric(block, m) for m in METRICS}


def build_table(root: Path) -> list[dict]:
    rows = []
    for dirname, block_header, label in VARIANTS:
        log_path = root / dirname / "training.log"
        metrics = parse_log(log_path, block_header)
        rows.append({"variant": label, **metrics})
    return rows


def format_table(rows: list[dict], root: Path, dataset: str, seed: str) -> str:
    col_w = 28
    met_w = 10
    lines = []
    lines.append(f"Ablation summary — dataset={dataset}  seed={seed}")
    lines.append(f"Root: {root}")
    lines.append("")
    header = f"{'Variant':<{col_w}}" + "".join(f"{m:>{met_w}}" for m in METRICS)
    lines.append(header)
    lines.append("-" * len(header))
    for row in rows:
        line = f"{row['variant']:<{col_w}}"
        line += "".join(f"{row[m]:>{met_w}}" for m in METRICS)
        lines.append(line)
    lines.append("")
    lines.append("Log files:")
    for dirname, _, _ in VARIANTS:
        log = root / dirname / "training.log"
        status = "OK" if log.exists() else "MISSING"
        lines.append(f"  [{status}] {log}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True,
                        help="Ablation root directory (e.g. ~/RulE/outputs/ablation_umls_seed42)")
    parser.add_argument("--dataset", default=None,
                        help="Dataset label for the header (inferred from root name if omitted)")
    parser.add_argument("--seed", default=None,
                        help="Seed label for the header (inferred from root name if omitted)")
    parser.add_argument("--no_write", action="store_true",
                        help="Print to stdout only; do not write SUMMARY.txt")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        print(f"ERROR: root directory does not exist: {root}", file=sys.stderr)
        sys.exit(1)

    # Infer dataset/seed from directory name like "ablation_umls_seed42"
    dataset = args.dataset
    seed = args.seed
    if dataset is None or seed is None:
        parts = root.name.split("_")
        if dataset is None:
            dataset = parts[1] if len(parts) > 1 else root.name
        if seed is None:
            seed_part = next((p for p in parts if p.startswith("seed")), None)
            seed = seed_part[4:] if seed_part else "?"

    rows = build_table(root)
    table = format_table(rows, root, dataset, seed)

    print(table)

    if not args.no_write:
        summary_path = root / "SUMMARY.txt"
        summary_path.write_text(table + "\n")
        print(f"\nWritten to: {summary_path}")


if __name__ == "__main__":
    main()
