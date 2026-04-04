"""
A/B Comparison: Original MLP vs New GAT Grounding.

Runs both methods from a shared pre-trained checkpoint and compares results.
All output is logged to comparison_results.txt with timing.

Usage:
    python run_comparison.py --config ../config/family_comparison.json
"""
import os
import sys
import re
import time
import shutil
import argparse
import subprocess


def run_stage(cmd, label, log_file, env=None, cwd=None):
    """Run a subprocess, stream output, and return elapsed time."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}\n")
    log_file.write(f"\n{'='*60}\n")
    log_file.write(f"  {label}\n")
    log_file.write(f"{'='*60}\n\n")
    log_file.flush()

    start = time.time()
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env=env, text=True, bufsize=1, cwd=cwd,
    )
    output_lines = []
    for line in proc.stdout:
        sys.stdout.write(line)
        log_file.write(line)
        output_lines.append(line)
    proc.wait()
    elapsed = time.time() - start

    status = "OK" if proc.returncode == 0 else f"FAILED (code {proc.returncode})"
    msg = f"[{label}] {status} in {elapsed:.1f}s\n"
    print(msg)
    log_file.write(msg)
    log_file.flush()

    return elapsed, output_lines, proc.returncode


def parse_metrics_from_log(log_path):
    """Extract the last set of test metrics from a training log file."""
    metrics = {}
    if not os.path.exists(log_path):
        return metrics

    with open(log_path) as f:
        lines = f.readlines()

    last_test_block = None
    for i, line in enumerate(lines):
        if "Evaluating on test" in line:
            last_test_block = i

    if last_test_block is None:
        return metrics

    for line in lines[last_test_block:]:
        for key in ["Hit1", "Hit3", "Hit10", "MR", "MRR"]:
            m = re.search(rf'{key}\s*:\s*([\d.]+)', line)
            if m:
                metrics[key] = float(m.group(1))
    return metrics


def parse_metrics_from_lines(lines):
    """Extract the last set of test metrics from captured output lines."""
    metrics = {}
    last_test_idx = None
    for i, line in enumerate(lines):
        if "Evaluating on test" in line:
            last_test_idx = i

    if last_test_idx is None:
        return metrics

    for line in lines[last_test_idx:]:
        for key in ["Hit1", "Hit3", "Hit10", "MR", "MRR"]:
            m = re.search(rf'{key}\s*:\s*([\d.]+)', line)
            if m:
                metrics[key] = float(m.group(1))
    return metrics


def main():
    parser = argparse.ArgumentParser(description="A/B Comparison: MLP vs GAT grounding")
    parser.add_argument('--config', type=str, required=True,
                        help='Path to shared config JSON (e.g. ../config/family_comparison.json)')
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    original_src = os.path.join(project_root, "RulE_original", "src")
    new_src = os.path.join(project_root, "src")
    outputs_dir = os.path.join(project_root, "outputs", "family_comparison")

    os.makedirs(outputs_dir, exist_ok=True)

    # Create a temporary config with absolute paths so both codebases resolve correctly
    import json
    with open(config_path) as f:
        config = json.load(f)

    config_dir = os.path.dirname(config_path)
    for key in ["data_path", "rule_file", "save_path"]:
        val = config.get(key)
        if val and not os.path.isabs(val):
            config[key] = os.path.normpath(os.path.join(config_dir, val))

    abs_config_path = os.path.join(outputs_dir, "comparison_config_abs.json")
    with open(abs_config_path, "w") as f:
        json.dump(config, f, indent=4)
    config_path = abs_config_path

    results_path = os.path.join(outputs_dir, "comparison_results.txt")
    results_file = open(results_path, "w")
    results_file.write("A/B Comparison: Original MLP vs New GAT Grounding\n")
    results_file.write(f"Config: {config_path}\n")
    results_file.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    results_file.flush()

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    total_start = time.time()

    # ── Stage 1 + 2A: Original method (pre-train + MLP grounding) ──
    original_cmd = [
        sys.executable, "-u",
        os.path.join(original_src, "main.py"),
        "-init", config_path,
    ]
    t_original, lines_original, rc_original = run_stage(
        original_cmd,
        "Stage 1+2A: Pre-training + Original MLP Grounding",
        results_file,
        env=env,
        cwd=original_src,
    )

    if rc_original != 0:
        results_file.write("\nORIGINAL RUN FAILED. Aborting comparison.\n")
        results_file.close()
        print(f"\nResults written to: {results_path}")
        return

    # Locate the pre-trained checkpoint produced by original run
    checkpoint_path = os.path.join(config["save_path"], "checkpoint")
    if not os.path.exists(checkpoint_path):
        results_file.write(f"\nERROR: Pre-trained checkpoint not found at {checkpoint_path}\n")
        results_file.close()
        print(f"\nResults written to: {results_path}")
        return

    # ── Stage 2B: New GAT grounding (skip pre-training, reuse checkpoint) ──
    gat_save_path = os.path.join(config["save_path"], "gat_grounding")
    os.makedirs(gat_save_path, exist_ok=True)

    gat_cmd = [
        sys.executable, "-u",
        os.path.join(new_src, "main.py"),
        "-init", config_path,
        "--skip_pretrain",
        "--pretrain_checkpoint", checkpoint_path,
        "-save", gat_save_path,
    ]
    t_gat, lines_gat, rc_gat = run_stage(
        gat_cmd,
        "Stage 2B: New GAT Grounding (shared pre-trained embeddings)",
        results_file,
        env=env,
        cwd=new_src,
    )

    # ── Parse results ──
    original_log = os.path.join(config["save_path"], "train.log")
    original_metrics = parse_metrics_from_log(original_log)
    if not original_metrics:
        original_metrics = parse_metrics_from_lines(lines_original)

    gat_log = os.path.join(gat_save_path, "train.log")
    gat_metrics = parse_metrics_from_log(gat_log)
    if not gat_metrics:
        gat_metrics = parse_metrics_from_lines(lines_gat)

    total_elapsed = time.time() - total_start

    # ── Write comparison ──
    results_file.write(f"\n{'='*60}\n")
    results_file.write("COMPARISON RESULTS\n")
    results_file.write(f"{'='*60}\n\n")

    results_file.write(f"{'Metric':<10} {'Original MLP':<15} {'New GAT':<15} {'Diff':<10}\n")
    results_file.write("-" * 50 + "\n")

    for key in ["Hit1", "Hit3", "Hit10", "MR", "MRR"]:
        orig_val = original_metrics.get(key, float('nan'))
        gat_val = gat_metrics.get(key, float('nan'))
        diff = gat_val - orig_val
        sign = "+" if diff >= 0 else ""
        results_file.write(f"{key:<10} {orig_val:<15.4f} {gat_val:<15.4f} {sign}{diff:<10.4f}\n")

    results_file.write(f"\n{'='*60}\n")
    results_file.write("TIMING\n")
    results_file.write(f"{'='*60}\n\n")
    results_file.write(f"Original (pre-train + MLP grounding): {t_original:.1f}s ({t_original/60:.1f} min)\n")
    results_file.write(f"GAT grounding only:                   {t_gat:.1f}s ({t_gat/60:.1f} min)\n")
    results_file.write(f"Total:                                {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)\n")
    results_file.write(f"\nFinished: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    results_file.close()

    # Also print summary to stdout
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}\n")
    print(f"{'Metric':<10} {'Original MLP':<15} {'New GAT':<15} {'Diff':<10}")
    print("-" * 50)
    for key in ["Hit1", "Hit3", "Hit10", "MR", "MRR"]:
        orig_val = original_metrics.get(key, float('nan'))
        gat_val = gat_metrics.get(key, float('nan'))
        diff = gat_val - orig_val
        sign = "+" if diff >= 0 else ""
        print(f"{key:<10} {orig_val:<15.4f} {gat_val:<15.4f} {sign}{diff:<10.4f}")

    print(f"\nTiming:")
    print(f"  Original (pre-train + MLP): {t_original/60:.1f} min")
    print(f"  GAT grounding only:         {t_gat/60:.1f} min")
    print(f"  Total:                      {total_elapsed/60:.1f} min")
    print(f"\nFull results saved to: {results_path}")


if __name__ == "__main__":
    main()
