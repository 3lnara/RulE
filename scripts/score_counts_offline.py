#!/usr/bin/env python3
"""
Offline binary-vs-raw x keep-vs-clamp MRR scoring from the pre-dumped counts.

All inputs come from dump_rule_counts.py; no model loading or grounding needed.

The scoring logic is a direct reimplementation of trainer.GroundTrainer.evaluate
(src_additive/trainer.py lines 573-681) using the four combinations of:
  basis : 'binary'  ->  1 if rule fired for entity, 0 otherwise
           'raw'     ->  actual integer path count
  clamp : True      ->  w_R = max(0, w_R_unclamped)
           False     ->  w_R = w_R_unclamped  (keep, possibly negative)

Ungrounded entities score 0 (matching simple_aggregation + no_bias, which sets
mask=ones so every entity competes; ungrounded ones get sum_R w_R * 0 = 0).

Outputs in --analysis_dir (default outputs/additive_umls_aggregation_2x2/analysis/):
  binary_vs_raw_<split>.csv    aggregate 2x2 table (one row per variant)
  ranks_<split>.csv            per-query (h, r, gold, L/H for each of 4 variants)

Prints the 2x2 MRR / Hit@1/3/10 table.

VALIDATION: binary+clamp valid MRR should match run.log -> 0.647622 (test 0.651140).

Usage (run from repo root):
    python scripts/score_counts_offline.py \\
        --analysis_dir outputs/additive_umls_aggregation_2x2/analysis \\
        --data_path    data/umls \\
        --splits       valid test
"""

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import torch


# ---------------------------------------------------------------------------
# hr2ooo filter: mirrors data.py KnowledgeGraph construction
# enc(h, r) = r * N + h  (same as data.py line 385)
# ---------------------------------------------------------------------------

def build_hr2ooo(data_path: str, N: int, num_rel: int) -> Dict[int, set]:
    """Build hr2ooo from all three splits + both directions."""
    hr2ooo: Dict[int, set] = defaultdict(set)

    def enc(h, r):
        return r * N + h

    for fname in ("train.txt", "valid.txt", "test.txt"):
        with open(os.path.join(data_path, fname)) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                # entity/relation IDs are already integers in .dict files,
                # but .txt files contain names -> use the dict files.
                pass
    # Use the .dict files to build name->id, then parse triples.
    ent2id = {}
    rel2id = {}
    with open(os.path.join(data_path, "entities.dict")) as f:
        for line in f:
            idx, name = line.strip().split("\t")
            ent2id[name] = int(idx)
    with open(os.path.join(data_path, "relations.dict")) as f:
        for line in f:
            idx, name = line.strip().split("\t")
            rel2id[name] = int(idx)

    for fname in ("train.txt", "valid.txt", "test.txt"):
        with open(os.path.join(data_path, fname)) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                h = ent2id[parts[0]]
                r = rel2id[parts[1]]
                t = ent2id[parts[2]]
                hr2ooo[enc(h, r)].add(t)
                hr2ooo[enc(t, r + num_rel)].add(h)

    return hr2ooo


# ---------------------------------------------------------------------------
# Core scoring helper (importable by analyze_binary_vs_raw_wins.py)
# ---------------------------------------------------------------------------

def score_queries(
    query_h: torch.Tensor,
    query_r: torch.Tensor,
    query_gold: torch.Tensor,
    q_ptr: torch.Tensor,
    rule_id: torch.Tensor,
    ent: torch.Tensor,
    cnt: torch.Tensor,
    w_R: torch.Tensor,
    basis: str,
    clamp: bool,
    hr2ooo: Dict[int, set],
    N: int,
) -> List[Tuple[int, int, int, int, int]]:
    """Score all queries and return per-query (h, r, gold, L, H) list.

    basis : 'binary' or 'raw'
    clamp : whether to zero negative w_R
    Returns list of (h, r, gold, L, H) — L/H are the filtered rank range.
    """
    w_eff = w_R.clamp(min=0.0) if clamp else w_R.clone()

    def enc(h, r):
        return r * N + h

    results = []
    Q = query_h.size(0)
    for qi in range(Q):
        h     = query_h[qi].item()
        r     = query_r[qi].item()
        gold  = query_gold[qi].item()
        start = q_ptr[qi].item()
        stop  = q_ptr[qi + 1].item()

        # Build score vector (float64 for precision)
        score = torch.zeros(N, dtype=torch.float64)
        if start < stop:
            r_ids  = rule_id[start:stop]
            ents   = ent[start:stop]
            cnts   = cnt[start:stop].long()
            w_vals = w_eff[r_ids].double()
            if basis == "binary":
                vals = w_vals            # weight * 1
            else:
                vals = w_vals * cnts.double()   # weight * count
            score.scatter_add_(0, ents, vals)

        # flag = True for entities NOT in hr2ooo (pure competitors).
        # Mirrors ValidDataset/TestDataset: mask starts all-True, then
        # ALL hr2ooo entries (including gold t) are set to False.
        # Gold is excluded from the competitor set -- same as the model eval.
        known_true = hr2ooo.get(enc(h, r), set())
        flag = torch.ones(N, dtype=torch.bool)
        for t in known_true:
            flag[t] = False   # includes gold itself

        val  = score[gold].item()
        n_gt = int((score[flag] > val).sum().item())
        n_ge = int((score[flag] >= val).sum().item())
        L    = n_gt + 1
        H    = n_ge + 2     # = (count >= val) + 2, matches trainer.evaluate

        results.append((h, r, gold, L, H))

    return results


def expectation_metrics(
    results: List[Tuple[int, int, int, int, int]]
) -> Dict[str, float]:
    """Replicate trainer.evaluate expectation over [L, H)."""
    h1 = h3 = h10 = mr = mrr = 0.0
    n = len(results)
    for (_, _, _, L, H) in results:
        span = H - L
        for rank in range(L, H):
            w = 1.0 / span
            if rank <= 1:  h1  += w
            if rank <= 3:  h3  += w
            if rank <= 10: h10 += w
            mr  += w * rank
            mrr += w / rank
    if n == 0:
        return {}
    return {
        "MRR":    mrr  / n,
        "MR":     mr   / n,
        "Hit@1":  h1   / n,
        "Hit@3":  h3   / n,
        "Hit@10": h10  / n,
        "n":      n,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

VARIANTS = [
    ("binary", False, "binary_keep"),
    ("binary", True,  "binary_clamp"),
    ("raw",    False, "raw_keep"),
    ("raw",    True,  "raw_clamp"),
]

# Per-dataset run.log MRRs to validate the offline scorer against.
# Each maps a model run -> the offline variant it should reproduce.
#   umls   : additive_umls_paper_sum_clamp  -> binary_clamp
#   family : paper_sum (binary, no clamp)   -> binary_keep
#            simple_aggregation_nobias (raw, no clamp) -> raw_keep
REFERENCES = {
    "umls": {
        "binary_clamp": {"valid": 0.647622, "test": 0.651140},
    },
    "family": {
        "binary_keep": {"valid": 0.933965, "test": 0.933928},
        "raw_keep":    {"valid": 0.870748, "test": 0.878099},
    },
}


def infer_dataset(data_path: str) -> str:
    base = os.path.basename(os.path.normpath(data_path)).lower()
    return base


def parse_args():
    p = argparse.ArgumentParser(
        description="Offline binary/raw x keep/clamp MRR from dumped counts.")
    p.add_argument("--analysis_dir",
                   default="outputs/additive_umls_aggregation_2x2/analysis")
    p.add_argument("--data_path",  default="data/umls")
    p.add_argument("--splits",     nargs="+", default=["valid", "test"])
    p.add_argument("--dataset", default=None,
                   help="Validation key into REFERENCES (default: inferred from "
                        "--data_path basename, e.g. 'umls', 'family').")
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve to repo root so relative paths work.
    _here = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.normpath(os.path.join(_here, ".."))
    os.chdir(_root)

    dataset = args.dataset or infer_dataset(args.data_path)
    print(f"Dataset for validation: '{dataset}'")

    # Load rule meta once (shared across splits)
    meta_path = os.path.join(args.analysis_dir, "rule_meta.pt")
    print(f"Loading rule_meta from {meta_path} ...")
    meta = torch.load(meta_path, weights_only=False)
    w_R_unclamped: torch.Tensor = meta["w_R_unclamped"].float()
    R = w_R_unclamped.size(0)
    print(f"  rules={R}  negative w_R={int((w_R_unclamped < 0).sum())}")

    for split in args.splits:
        counts_path = os.path.join(args.analysis_dir, f"counts_{split}.pt")
        print(f"\nLoading {counts_path} ...")
        counts = torch.load(counts_path, weights_only=False)
        query_h    = counts["query_h"]
        query_r    = counts["query_r"]
        query_gold = counts["query_gold"]
        q_ptr      = counts["q_ptr"]
        rule_id    = counts["rule_id"]
        ent        = counts["ent"]
        cnt        = counts["cnt"]
        Q          = query_h.size(0)
        N          = int(query_h.max().item()) + 1  # will be overridden below

        # Infer N properly from entity dict
        ent2id = {}
        with open(os.path.join(args.data_path, "entities.dict")) as f:
            for line in f:
                idx, name = line.strip().split("\t")
                ent2id[name] = int(idx)
        N = len(ent2id)
        rel2id = {}
        with open(os.path.join(args.data_path, "relations.dict")) as f:
            for line in f:
                idx, name = line.strip().split("\t")
                rel2id[name] = int(idx)
        num_rel = len(rel2id)

        print(f"  queries={Q}  N={N}  nnz={rule_id.size(0)}")

        print("Building hr2ooo filter...")
        hr2ooo = build_hr2ooo(args.data_path, N, num_rel)

        # Score all 4 variants
        all_results = {}
        for basis, clamp, label in VARIANTS:
            print(f"  scoring {label} ...")
            res = score_queries(
                query_h, query_r, query_gold,
                q_ptr, rule_id, ent, cnt,
                w_R_unclamped, basis, clamp, hr2ooo, N,
            )
            all_results[label] = res

        # Aggregate metrics table
        print(f"\n  === {split.upper()} 2x2 results ===")
        header = f"{'variant':<16}{'MRR':>9}{'Hit@1':>9}{'Hit@3':>9}{'Hit@10':>9}{'MR':>8}"
        print(header)
        print("-" * len(header))
        agg_rows = []
        for _, _, label in VARIANTS:
            m = expectation_metrics(all_results[label])
            print(f"{label:<16}{m['MRR']:>9.6f}{m['Hit@1']:>9.6f}"
                  f"{m['Hit@3']:>9.6f}{m['Hit@10']:>9.6f}{m['MR']:>8.2f}")
            agg_rows.append({
                "split": split, "variant": label,
                "MRR":    round(m["MRR"],    6),
                "Hit@1":  round(m["Hit@1"],  6),
                "Hit@3":  round(m["Hit@3"],  6),
                "Hit@10": round(m["Hit@10"], 6),
                "MR":     round(m["MR"],     4),
                "n":      m["n"],
            })

        # Validate against known run.log values for this dataset (every
        # variant/split with a registered reference).
        ds_refs = REFERENCES.get(dataset, {})
        printed_header = False
        for variant, split_refs in ds_refs.items():
            ref = split_refs.get(split)
            if ref is None:
                continue
            if not printed_header:
                print(f"\n  Validation vs run.log (dataset='{dataset}'):")
                printed_header = True
            got = next(r["MRR"] for r in agg_rows if r["variant"] == variant)
            delta = abs(got - ref)
            status = "PASS" if delta < 1e-4 else "FAIL"
            print(f"    {variant:<14} got={got:.6f}  ref={ref:.6f}  "
                  f"delta={delta:.2e}  [{status}]")

        # Write aggregate CSV
        agg_csv = os.path.join(args.analysis_dir, f"binary_vs_raw_{split}.csv")
        with open(agg_csv, "w", newline="") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["split","variant","MRR","Hit@1","Hit@3","Hit@10","MR","n"])
            writer.writeheader()
            writer.writerows(agg_rows)
        print(f"\n  Wrote {agg_csv}")

        # Write per-query ranks CSV
        ranks_csv = os.path.join(args.analysis_dir, f"ranks_{split}.csv")
        fieldnames = ["h", "r", "gold"] + [
            f"L_{v}" for *_, v in VARIANTS
        ] + [
            f"H_{v}" for *_, v in VARIANTS
        ]
        with open(ranks_csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            # Zip results across variants for each query
            zipped = list(zip(
                *[all_results[label] for _, _, label in VARIANTS]
            ))
            for qi, per_variant in enumerate(zipped):
                # per_variant: tuple of (h, r, gold, L, H) for each of 4 variants
                h_val, r_val, g_val = per_variant[0][:3]
                row = {"h": h_val, "r": r_val, "gold": g_val}
                for (_, _, label), (_, _, _, L, H) in zip(VARIANTS, per_variant):
                    row[f"L_{label}"] = L
                    row[f"H_{label}"] = H
                writer.writerow(row)
        print(f"  Wrote {ranks_csv}  ({Q} rows)")

    print("\nDone.")


if __name__ == "__main__":
    main()
