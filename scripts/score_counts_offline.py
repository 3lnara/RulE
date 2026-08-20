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
  binary_vs_raw_<split>.csv       aggregate 2x2 table (confidence weighting)
  ranks_<split>.csv               per-query L/H for each of 4 confidence variants
  precision_binary_<split>.csv    aggregate table when --weight_source precision

Prints the MRR / Hit@1/3/10 table.

VALIDATION: binary+clamp valid MRR should match run.log -> 0.647622 (test 0.651140).

--weight_source precision mode
  Weights: per-rule PCA precision from rule_precision.pt (written by
           rule_precision_train.py).  Precision is in [0,1], so no clamping
           is needed (binary_keep == binary_clamp).  NaN precision values
           (rules that never fired from a known subject during training) are
           treated as weight 0 so they contribute nothing to scores.
  The resulting MRR is directly comparable to the confidence binary_clamp
  row in binary_vs_raw_<split>.csv.

Usage (from repo root):
    # Confidence mode (default, reproduces existing results):
    python scripts/score_counts_offline.py \\
        --analysis_dir outputs/additive_umls_aggregation_2x2/analysis \\
        --data_path    data/umls \\
        --splits       valid test

    # Precision mode (requires rule_precision.pt in --analysis_dir):
    python scripts/score_counts_offline.py \\
        --analysis_dir outputs/additive_umls_aggregation_2x2/analysis \\
        --data_path    data/umls \\
        --weight_source precision \\
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
    p.add_argument("--weight_source", choices=["confidence", "precision", "uniform"],
                   default="confidence",
                   help="Weight source for scoring. 'confidence' (default): uses "
                        "w_R_unclamped from rule_meta.pt, runs all 4 variants "
                        "(binary/raw x keep/clamp). 'precision': uses per-rule "
                        "PCA precision from rule_precision.pt, runs binary only "
                        "(precision >= 0 so clamping is a no-op). 'uniform': sets "
                        "every rule weight to 1.0 (unweighted baseline = count of "
                        "distinct rules that fired per entity), runs binary only.")
    p.add_argument("--precision_file", default=None,
                   help="Path to rule_precision.pt. Defaults to "
                        "<analysis_dir>/rule_precision.pt when --weight_source precision.")
    return p.parse_args()


def _load_entity_rel_sizes(data_path: str):
    """Return (N, num_rel) from entities.dict and relations.dict."""
    ent2id = {}
    with open(os.path.join(data_path, "entities.dict")) as f:
        for line in f:
            idx, name = line.strip().split("\t")
            ent2id[name] = int(idx)
    rel2id = {}
    with open(os.path.join(data_path, "relations.dict")) as f:
        for line in f:
            idx, name = line.strip().split("\t")
            rel2id[name] = int(idx)
    return len(ent2id), len(rel2id)


def main():
    args = parse_args()

    # Resolve to repo root so relative paths work.
    _here = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.normpath(os.path.join(_here, ".."))
    os.chdir(_root)

    dataset = args.dataset or infer_dataset(args.data_path)
    print(f"Dataset       : '{dataset}'")
    print(f"Weight source : {args.weight_source}")

    # ---- Load rule meta (always needed for alignment check in precision mode) ----
    meta_path = os.path.join(args.analysis_dir, "rule_meta.pt")
    print(f"Loading rule_meta from {meta_path} ...")
    try:
        meta = torch.load(meta_path, weights_only=False)
    except TypeError:
        # PyTorch < 1.13 (e.g. local rule_env on 1.11) has no weights_only kwarg
        meta = torch.load(meta_path)
    w_R_unclamped: torch.Tensor = meta["w_R_unclamped"].float()
    R = w_R_unclamped.size(0)
    print(f"  rules={R}  negative w_R={int((w_R_unclamped < 0).sum())}")

    # ---- Resolve scoring weights by source -------------------------------------
    # variant_label: row name in the printed table + output CSV stem.
    if args.weight_source == "precision":
        prec_path = args.precision_file or os.path.join(args.analysis_dir, "rule_precision.pt")
        print(f"Loading precision weights from {prec_path} ...")
        try:
            prec_data = torch.load(prec_path, weights_only=False)
        except TypeError:
            prec_data = torch.load(prec_path)   # torch < 1.13 has no weights_only
        prec_tensor: torch.Tensor = prec_data["precision"].float()
        if prec_tensor.size(0) != R:
            raise ValueError(
                f"rule_precision.pt has {prec_tensor.size(0)} entries but "
                f"rule_meta.pt has {R} rules. "
                "Ensure both come from the same training run and dataset.")
        # NaN -> 0.0: rules that never fired from a known subject contribute nothing.
        n_nan = int(torch.isnan(prec_tensor).sum().item())
        n_pos = int((prec_tensor > 0).sum().item())
        prec_tensor = torch.nan_to_num(prec_tensor, nan=0.0)
        print(f"  precision range=[{prec_tensor.min():.4f}, {prec_tensor.max():.4f}]  "
              f"  rules with precision>0: {n_pos}/{R}  nan->0: {n_nan}")
        w_for_scoring = prec_tensor
        variant_label = "precision_binary"
    elif args.weight_source == "uniform":
        # Every rule weight = 1.0: score[t] = #{distinct rules that fired on t}.
        w_for_scoring = torch.ones(R, dtype=torch.float32)
        variant_label = "uniform_binary"
        print(f"  uniform weights: all {R} rules set to w_R=1.0 "
              "(unweighted rule-count baseline)")
    else:
        w_for_scoring = w_R_unclamped
        variant_label = None  # confidence mode runs the 4-variant table

    for split in args.splits:
        counts_path = os.path.join(args.analysis_dir, f"counts_{split}.pt")
        print(f"\nLoading {counts_path} ...")
        try:
            counts = torch.load(counts_path, weights_only=False)
        except TypeError:
            counts = torch.load(counts_path)   # torch < 1.13 has no weights_only
        query_h    = counts["query_h"]
        query_r    = counts["query_r"]
        query_gold = counts["query_gold"]
        q_ptr      = counts["q_ptr"]
        rule_id    = counts["rule_id"]
        ent        = counts["ent"]
        cnt        = counts["cnt"]
        Q          = query_h.size(0)

        N, num_rel = _load_entity_rel_sizes(args.data_path)
        print(f"  queries={Q}  N={N}  nnz={rule_id.size(0)}")

        print("Building hr2ooo filter...")
        hr2ooo = build_hr2ooo(args.data_path, N, num_rel)

        if args.weight_source != "confidence":
            # precision / uniform: weights are >= 0, so clamping is a no-op and
            # raw vs binary differ only by count magnitude -> run binary only.
            print(f"  scoring {variant_label} ...")
            res_var = score_queries(
                query_h, query_r, query_gold,
                q_ptr, rule_id, ent, cnt,
                w_for_scoring, "binary", False, hr2ooo, N,
            )
            m_var = expectation_metrics(res_var)

            # Gather reference MRRs for side-by-side comparison.
            import csv as _csv_mod
            refs = []  # list of (label, MRR)

            conf_csv = os.path.join(args.analysis_dir, f"binary_vs_raw_{split}.csv")
            if os.path.exists(conf_csv):
                with open(conf_csv) as fh:
                    for row in _csv_mod.DictReader(fh):
                        if row.get("variant") == "binary_clamp":
                            refs.append(("confidence_binary_clamp", float(row["MRR"])))
                            break

            # When running uniform, also show the precision_binary result if present.
            if args.weight_source == "uniform":
                prec_csv = os.path.join(args.analysis_dir, f"precision_binary_{split}.csv")
                if os.path.exists(prec_csv):
                    with open(prec_csv) as fh:
                        for row in _csv_mod.DictReader(fh):
                            if row.get("variant") == "precision_binary":
                                refs.append(("precision_binary", float(row["MRR"])))
                                break

            print(f"\n  === {split.upper()} {variant_label} ===")
            hdr = f"{'variant':<26}{'MRR':>9}{'Hit@1':>9}{'Hit@3':>9}{'Hit@10':>9}{'MR':>8}"
            print(hdr)
            print("-" * len(hdr))
            print(f"{variant_label:<26}"
                  f"{m_var['MRR']:>9.6f}{m_var['Hit@1']:>9.6f}"
                  f"{m_var['Hit@3']:>9.6f}{m_var['Hit@10']:>9.6f}"
                  f"{m_var['MR']:>8.2f}")
            for ref_label, ref_mrr in refs:
                print(f"{ref_label:<26}{ref_mrr:>9.6f}"
                      f"{'(reference)':>35}")
            for ref_label, ref_mrr in refs:
                delta = m_var["MRR"] - ref_mrr
                print(f"  Delta {variant_label} - {ref_label}: {delta:+.6f}")

            # Write <variant_label>_<split>.csv
            agg_csv = os.path.join(args.analysis_dir, f"{variant_label}_{split}.csv")
            agg_rows = [{
                "split":   split,
                "variant": variant_label,
                "MRR":    round(m_var["MRR"],    6),
                "Hit@1":  round(m_var["Hit@1"],  6),
                "Hit@3":  round(m_var["Hit@3"],  6),
                "Hit@10": round(m_var["Hit@10"], 6),
                "MR":     round(m_var["MR"],     4),
                "n":      m_var["n"],
            }]
            with open(agg_csv, "w", newline="") as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=["split", "variant", "MRR",
                                "Hit@1", "Hit@3", "Hit@10", "MR", "n"])
                writer.writeheader()
                writer.writerows(agg_rows)
            print(f"\n  Wrote {agg_csv}")

        else:
            # ---- Confidence mode: original 4-variant 2x2 ----------------------
            all_results = {}
            for basis, clamp, label in VARIANTS:
                print(f"  scoring {label} ...")
                res = score_queries(
                    query_h, query_r, query_gold,
                    q_ptr, rule_id, ent, cnt,
                    w_for_scoring, basis, clamp, hr2ooo, N,
                )
                all_results[label] = res

            print(f"\n  === {split.upper()} 2x2 results ===")
            header = (f"{'variant':<16}{'MRR':>9}{'Hit@1':>9}"
                      f"{'Hit@3':>9}{'Hit@10':>9}{'MR':>8}")
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

            # Validate against known run.log MRRs.
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
                    fh,
                    fieldnames=["split", "variant", "MRR",
                                "Hit@1", "Hit@3", "Hit@10", "MR", "n"])
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
                zipped = list(zip(*[all_results[label] for _, _, label in VARIANTS]))
                for qi, per_variant in enumerate(zipped):
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
