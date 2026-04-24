#!/usr/bin/env python3
"""Summarize unified evaluator JSON files into one comparison table."""

import argparse
import json
import os
from statistics import mean, pstdev


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment_dir", required=True, type=str)
    p.add_argument("--output_json", required=True, type=str)
    return p.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_metric(results, metric_name):
    vals = [r[metric_name] for r in results if r is not None and metric_name in r]
    if not vals:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": mean(vals), "std": pstdev(vals) if len(vals) > 1 else 0.0, "n": len(vals)}


def main():
    args = parse_args()
    eval_dir = os.path.join(args.experiment_dir, "evaluations")
    if not os.path.isdir(eval_dir):
        raise FileNotFoundError(f"Missing evaluations directory: {eval_dir}")

    per_seed = []
    for seed_dir in sorted(os.listdir(eval_dir)):
        seed_path = os.path.join(eval_dir, seed_dir)
        if not os.path.isdir(seed_path):
            continue
        original_path = os.path.join(seed_path, "original_eval.json")
        gat_path = os.path.join(seed_path, "gat_eval.json")
        if not (os.path.exists(original_path) and os.path.exists(gat_path)):
            continue
        per_seed.append(
            {
                "seed": seed_dir,
                "original": load_json(original_path),
                "gat": load_json(gat_path),
            }
        )

    def pack(name, fixed_rows, adaptive_rows):
        return {
            "variant": name,
            "fixed": {
                "MRR": collect_metric(fixed_rows, "MRR"),
                "Hit@1": collect_metric(fixed_rows, "Hit@1"),
                "Hit@3": collect_metric(fixed_rows, "Hit@3"),
                "Hit@10": collect_metric(fixed_rows, "Hit@10"),
            },
            "adaptive": {
                "MRR": collect_metric(adaptive_rows, "MRR"),
                "Hit@1": collect_metric(adaptive_rows, "Hit@1"),
                "Hit@3": collect_metric(adaptive_rows, "Hit@3"),
                "Hit@10": collect_metric(adaptive_rows, "Hit@10"),
            },
        }

    original_fixed = [r["original"]["fixed_metrics"] for r in per_seed]
    original_adaptive = [r["original"]["adaptive_metrics"] for r in per_seed]
    gat_fixed = [r["gat"]["fixed_metrics"] for r in per_seed]
    gat_adaptive = [r["gat"]["adaptive_metrics"] for r in per_seed]

    summary = {
        "num_seeds": len(per_seed),
        "variants": [
            pack("original", original_fixed, original_adaptive),
            pack("gat", gat_fixed, gat_adaptive),
        ],
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
