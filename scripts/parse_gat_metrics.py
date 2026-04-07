"""
Parse a GAT grounding run.log and produce:
  1. Per-iteration metrics table (valid set): timing, Hit1/3/10, MR, MRR
  2. Final evaluation table (valid / test / test_kge sets)
  3. Peak GPU memory per iteration (if nvidia-smi dmon log is provided)

Usage:
    python parse_gat_metrics.py --log <save_path>/run.log [--gpu_log <path>] [--out <path>]

Outputs:
    Prints tables to stdout and writes them to <out> (default: metrics_table.txt
    in the same directory as --log).
"""

import re
import argparse
import os
from datetime import datetime


LOG_TS_FMT = "%Y-%m-%d %H:%M:%S"

# Patterns
RE_TS       = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
RE_ITER     = re.compile(r"\| Iteration:\s*(\d+)/(\d+)")
RE_EVAL     = re.compile(r">>>>> Predictor: Evaluating on (\S+)")
RE_HIT1     = re.compile(r"Hit1\s*:\s*([\d.]+)")
RE_HIT3     = re.compile(r"Hit3\s*:\s*([\d.]+)")
RE_HIT10    = re.compile(r"Hit10\s*:\s*([\d.]+)")
RE_MR       = re.compile(r"MR\s*:\s*([\d.]+)")
RE_MRR      = re.compile(r"MRR\s*:\s*([\d.]+)")
RE_FINAL    = re.compile(r"\| Final Test MRR:\s*([\d.]+)")

# nvidia-smi dmon -s mu output: space-separated, column "fb" = used mem (MiB)
RE_DMON     = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)")   # gpu, sm%, mem%
RE_DMON_HDR = re.compile(r"#\s*gpu")


def parse_log(log_path):
    """Return (iter_records, final_records, final_mrr).

    iter_records: list of dicts, one per grounding iteration
        {iter, n_iters, start_ts, end_ts, wall_s, split, Hit1, Hit3, Hit10, MR, MRR}

    final_records: list of dicts for final evaluations (valid/test/test_kge)
        {split, Hit1, Hit3, Hit10, MR, MRR}
    """
    with open(log_path) as f:
        lines = f.readlines()

    iter_records = []
    final_records = []
    final_mrr = None

    # State machine
    current_iter = None
    current_iter_start = None
    n_iters = None
    in_final_phase = False      # after "Final Test MRR" line, evaluations are final
    current_eval_split = None
    current_eval_ts = None
    current_eval_is_iter_valid = False
    pending_metrics = {}

    def flush_metrics(split, ts):
        """Flush accumulated metrics for the completed evaluation block."""
        nonlocal current_iter, current_iter_start, iter_records, final_records

        m = dict(pending_metrics)
        pending_metrics.clear()

        if split is None:
            return

        if not in_final_phase and split == "valid" and current_iter is not None:
            # Per-iteration valid evaluation
            wall = (ts - current_iter_start).total_seconds() if current_iter_start else float("nan")
            iter_records.append({
                "iter":   current_iter,
                "n_iters": n_iters,
                "start_ts": current_iter_start,
                "end_ts":   ts,
                "wall_s":   wall,
                "split":    split,
                **m,
            })
        else:
            # Final evaluation (valid/test/test_kge at the end)
            final_records.append({"split": split, **m})

    i = 0
    while i < len(lines):
        line = lines[i]
        ts_match = RE_TS.match(line)
        ts = datetime.strptime(ts_match.group(1), LOG_TS_FMT) if ts_match else None

        # --- Iteration header ---
        m = RE_ITER.search(line)
        if m:
            # Flush any open eval block from previous iteration before starting new one
            if current_eval_split is not None and pending_metrics:
                flush_metrics(current_eval_split, current_eval_ts or ts)
                current_eval_split = None
                pending_metrics.clear()

            current_iter = int(m.group(1))
            n_iters = int(m.group(2))
            current_iter_start = ts
            i += 1
            continue

        # --- Final MRR marker ---
        m = RE_FINAL.search(line)
        if m:
            final_mrr = float(m.group(1))
            # Flush any open eval block
            if current_eval_split is not None and pending_metrics:
                flush_metrics(current_eval_split, current_eval_ts or ts)
                current_eval_split = None
                pending_metrics.clear()
            in_final_phase = True
            i += 1
            continue

        # --- Evaluate header ---
        m = RE_EVAL.search(line)
        if m:
            # Flush previous eval block if still open
            if current_eval_split is not None and pending_metrics:
                flush_metrics(current_eval_split, current_eval_ts or ts)
                pending_metrics.clear()
            current_eval_split = m.group(1)
            current_eval_ts = ts
            i += 1
            continue

        # --- Metric lines ---
        for pattern, key in [
            (RE_HIT1,  "Hit1"),
            (RE_HIT3,  "Hit3"),
            (RE_HIT10, "Hit10"),
            (RE_MR,    "MR"),
            (RE_MRR,   "MRR"),
        ]:
            mm = pattern.search(line)
            if mm:
                pending_metrics[key] = float(mm.group(1))
                # When MRR arrives, the block is complete — flush it
                if key == "MRR":
                    flush_metrics(current_eval_split, ts)
                    current_eval_split = None
                    pending_metrics.clear()
                break

        i += 1

    # Flush anything still open at EOF
    if current_eval_split is not None and pending_metrics:
        flush_metrics(current_eval_split, current_eval_ts)

    return iter_records, final_records, final_mrr


# def parse_gpu_log(gpu_log_path):
#     """Parse nvidia-smi dmon output and return list of (timestamp_str, fb_mib) tuples.

#     dmon -s mu -d 5 --id 0  output columns:
#       # gpu   sm  mem  enc  dec
#          0    xx   xx   xx   xx
#     dmon -s m  output (memory only):
#       # gpu    fb  bar1
#          0   xxxx  xxx

#     We detect the header to figure out which column is 'fb' (framebuffer used MiB).
#     """
#     if not gpu_log_path or not os.path.exists(gpu_log_path):
#         return []

#     samples = []
#     fb_col = None

#     with open(gpu_log_path) as f:
#         for line in f:
#             line = line.strip()
#             if not line or line.startswith("#"):
#                 # Try to detect column positions from header
#                 if "fb" in line.lower():
#                     headers = line.lstrip("# ").split()
#                     try:
#                         fb_col = headers.index("fb")
#                     except ValueError:
#                         pass
#                 continue
#             parts = line.split()
#             if len(parts) < 2:
#                 continue
#             try:
#                 if fb_col is not None and fb_col < len(parts):
#                     fb = int(parts[fb_col])
#                 else:
#                     # fallback: second column
#                     fb = int(parts[1])
#                 samples.append(fb)
#             except ValueError:
#                 continue

#     return samples


def fmt_time(seconds):
    if seconds != seconds:   # nan
        return "  N/A  "
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"


def build_iter_table(iter_records, gpu_samples=None, n_iters=None):
    """Return a formatted string table of per-iteration metrics."""
    if not iter_records:
        return "(no per-iteration records found)\n"

    # Assign GPU memory peaks to iterations if samples are available
    # We distribute samples uniformly across iterations (best-effort)
    gpu_peaks = {}
    if gpu_samples and len(gpu_samples) > 0 and iter_records:
        total = len(iter_records)
        chunk = max(1, len(gpu_samples) // total)
        for idx, rec in enumerate(iter_records):
            start = idx * chunk
            end = (idx + 1) * chunk if idx < total - 1 else len(gpu_samples)
            chunk_samples = gpu_samples[start:end]
            gpu_peaks[rec["iter"]] = max(chunk_samples) if chunk_samples else None

    has_gpu = bool(gpu_peaks)
    col_w = {
        "iter":   6,
        "wall":   10,
        "gpu":    10,
        "hit1":    8,
        "hit3":    8,
        "hit10":   8,
        "mr":     10,
        "mrr":     8,
    }

    header_parts = [
        f"{'Iter':>{col_w['iter']}}",
        f"{'Wall':>{col_w['wall']}}",
    ]
    if has_gpu:
        header_parts.append(f"{'GPU(MiB)':>{col_w['gpu']}}")
    header_parts += [
        f"{'Hit@1':>{col_w['hit1']}}",
        f"{'Hit@3':>{col_w['hit3']}}",
        f"{'Hit@10':>{col_w['hit10']}}",
        f"{'MR':>{col_w['mr']}}",
        f"{'MRR':>{col_w['mrr']}}",
    ]
    header = "  ".join(header_parts)
    sep = "-" * len(header)

    lines = [header, sep]
    for rec in iter_records:
        it = rec["iter"]
        parts = [
            f"{it:>{col_w['iter']}}",
            f"{fmt_time(rec.get('wall_s', float('nan'))):>{col_w['wall']}}",
        ]
        if has_gpu:
            peak = gpu_peaks.get(it)
            gpu_str = f"{peak}" if peak is not None else "N/A"
            parts.append(f"{gpu_str:>{col_w['gpu']}}")
        parts += [
            f"{rec.get('Hit1', float('nan')):.4f}".rjust(col_w["hit1"]),
            f"{rec.get('Hit3', float('nan')):.4f}".rjust(col_w["hit3"]),
            f"{rec.get('Hit10', float('nan')):.4f}".rjust(col_w["hit10"]),
            f"{rec.get('MR', float('nan')):.2f}".rjust(col_w["mr"]),
            f"{rec.get('MRR', float('nan')):.4f}".rjust(col_w["mrr"]),
        ]
        lines.append("  ".join(parts))

    lines.append(sep)
    total_wall = sum(r.get("wall_s", 0) for r in iter_records if r.get("wall_s") == r.get("wall_s"))
    lines.append(f"  Total wall time: {fmt_time(total_wall)}")

    if has_gpu and gpu_samples:
        lines.append(f"  Peak GPU memory (all iters): {max(gpu_samples)} MiB")

    return "\n".join(lines)


def build_final_table(final_records):
    """Return a formatted string table of final evaluation metrics."""
    if not final_records:
        return "(no final evaluation records found)\n"

    col_w = {"split": 12, "hit1": 8, "hit3": 8, "hit10": 8, "mr": 10, "mrr": 8}
    header = (
        f"{'Split':<{col_w['split']}}  "
        f"{'Hit@1':>{col_w['hit1']}}  "
        f"{'Hit@3':>{col_w['hit3']}}  "
        f"{'Hit@10':>{col_w['hit10']}}  "
        f"{'MR':>{col_w['mr']}}  "
        f"{'MRR':>{col_w['mrr']}}"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for rec in final_records:
        lines.append(
            f"{rec.get('split','?'):<{col_w['split']}}  "
            f"{rec.get('Hit1', float('nan')):.4f}".rjust(col_w["hit1"] + 2) +
            f"  {rec.get('Hit3', float('nan')):.4f}".rjust(col_w["hit3"] + 2) +
            f"  {rec.get('Hit10', float('nan')):.4f}".rjust(col_w["hit10"] + 2) +
            f"  {rec.get('MR', float('nan')):.2f}".rjust(col_w["mr"] + 2) +
            f"  {rec.get('MRR', float('nan')):.4f}".rjust(col_w["mrr"] + 2)
        )
    lines.append(sep)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Parse GAT grounding run.log into metric tables")
    parser.add_argument("--log",     required=True, help="Path to run.log")
    # parser.add_argument("--gpu_log", default=None,  help="Path to nvidia-smi dmon output file")
    parser.add_argument("--out",     default=None,  help="Output .txt path (default: same dir as --log)")
    args = parser.parse_args()

    if not os.path.exists(args.log):
        print(f"ERROR: log file not found: {args.log}")
        raise SystemExit(1)

    out_path = args.out or os.path.join(os.path.dirname(args.log), "metrics_table.txt")

    iter_records, final_records, final_mrr = parse_log(args.log)
    # gpu_samples = parse_gpu_log(args.gpu_log)

    n_iters = iter_records[0]["n_iters"] if iter_records else "?"

    sections = []
    sections.append(f"GAT Grounding Metrics  |  family dataset  |  {len(iter_records)}/{n_iters} iterations parsed")
    sections.append("=" * 70)

    # sections.append("\nPER-ITERATION VALID METRICS")
    # sections.append("-" * 70)
    # sections.append(build_iter_table(iter_records, gpu_samples, n_iters))

    sections.append("\nFINAL EVALUATION METRICS  (best checkpoint)")
    sections.append("-" * 70)
    sections.append(build_final_table(final_records))

    if final_mrr is not None:
        sections.append(f"\nBest valid MRR achieved: {final_mrr:.6f}")

    output = "\n".join(sections) + "\n"

    print(output)
    with open(out_path, "w") as f:
        f.write(output)
    print(f"\nTable written to: {out_path}")


if __name__ == "__main__":
    main()
