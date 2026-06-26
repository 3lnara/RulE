#!/usr/bin/env python3
"""
Offline MLP scoring from pre-dumped counts (Route B).

Reuses the trained original-RulE grounding checkpoint (outputs/umls/grounding.pt)
and pre-dumped counts (counts_<split>.pt) to produce per-query MLP ranks without
re-grounding or torch_scatter.

Scoring mirrors RulE_original/src/model.py forward() lines 559-577:

  For grounded entity e:
    score[e] = rule_to_entity(rule_count, rules_weight_emb[fired],
                              mlp_feature[fired])[e_idx]  +  bias[e]
  For ungrounded entity e:
    score[e] = bias[e]   (forward line 535 / 572 — bias added to ALL entities)

The grounding mask returned by forward() is always ones.bool(), so all
entities participate in filtered ranking (no special ungrounded path).

Validation: rule-only MRR must reproduce run.log within tol=2e-3:
  valid  MRR = 0.806168
  test   MRR = 0.805089

Outputs in --analysis_dir:
  ranks_mlp_<split>.csv   h, r, gold, L_mlp, H_mlp (per query)

Usage (from repo root):
    python scripts/score_mlp_offline.py \\
        --analysis_dir outputs/additive_umls_aggregation_2x2/analysis \\
        --checkpoint   outputs/umls/grounding.pt \\
        --data_path    data/umls \\
        --rule_file    data/umls/mined_rules.txt \\
        --splits       valid test
"""

import argparse
import csv
import os
import sys
import time

import torch

# ---------------------------------------------------------------------------
# Bootstrap: add RulE_original/src to sys.path for model.py + layers.py.
# data.py (which needs torch_scatter) is NOT imported.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.normpath(os.path.join(_SCRIPT_DIR, ".."))
_ORIG_SRC   = os.path.join(_REPO_ROOT, "RulE_original", "src")
if _ORIG_SRC not in sys.path:
    sys.path.insert(0, _ORIG_SRC)

from model import RulE  # noqa: E402  (layers.py imported transitively)

# Reuse scoring utilities from the sibling offline scorer.
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
from score_counts_offline import (  # noqa: E402
    build_hr2ooo,
    expectation_metrics,
    _load_entity_rel_sizes,
)

# ---------------------------------------------------------------------------
# Rule loading (identical to dump_rule_counts.py)
# ---------------------------------------------------------------------------

def load_rules(rule_file: str):
    """Load mined_rules.txt -> list of [global_id, r_head, b1, ...] (no padding)."""
    rules = []
    with open(rule_file) as f:
        for idx, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            rules.append([idx] + [int(p) for p in parts])
    return rules


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_and_load_model(checkpoint_path: str, rule_file: str,
                         N: int, num_rel: int,
                         device: torch.device):
    """Build RulE, set_rules (BEFORE load_state_dict), load checkpoint."""

    class _FakeGraph:
        """Exposes only entity_size / relation_size used by RulE.__init__."""
        entity_size   = N
        relation_size = num_rel

    rules = load_rules(rule_file)
    print(f"  {len(rules)} rules loaded from {rule_file}")

    model = RulE(
        graph=_FakeGraph(),
        p_norm=2,
        mlp_rule_dim=100,
        gamma_fact=6.0,
        gamma_rule=8.0,
        hidden_dim=2000,
        device=device,
        dataset="umls",   # -> score_model = MLP([100, 1]) for UMLS
    )

    # set_rules BEFORE load_state_dict: allocates mlp_feature/rule_emb with
    # correct shape so the checkpoint's trained weights are loaded into them.
    model.set_rules(rules)
    print(f"  num_rules={model.num_rules}  max_body_len={model.max_length}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Missing keys are fine (beta, feature_mean/std etc. not in checkpoint).
    print(f"  checkpoint loaded: missing={len(missing)}  unexpected={len(unexpected)}")
    if unexpected:
        print(f"  unexpected keys (first 5): {unexpected[:5]}")

    model.eval()
    model.to(device)

    print("  computing rules_weight_emb (eval_compute_rule_weight) ...")
    with torch.no_grad():
        # Guard against R < batch (not the case for UMLS but safe in general).
        orig_batch = 128
        R = model.rule_features.size(0)
        split_num = max(1, R // orig_batch)
        model.rule_masks    = model.rule_masks.to(device)
        model.rule_features = model.rule_features.to(device)
        rule_batches      = torch.split(model.rule_features, split_num, 0)
        rule_mask_batches = torch.split(model.rule_masks,    split_num, 0)
        embs = []
        for rules_b, masks_b in zip(rule_batches, rule_mask_batches):
            emb = model.add_ruleE_g(rules_b.unsqueeze(1), masks_b).squeeze(1)
            embs.append(emb)
        model.rules_weight_emb = torch.cat(embs, dim=0)
    print(f"  rules_weight_emb shape: {model.rules_weight_emb.shape}")
    return model, rules


# ---------------------------------------------------------------------------
# Alignment check
# ---------------------------------------------------------------------------

def alignment_check(model: RulE, meta: dict):
    """Verify rule count and head/length match between model and rule_meta.pt.

    Uses model.relation2rules (same approach as dump_rule_counts.py) to get
    actual body lengths, avoiding the ambiguity where padding_index == 46
    equals a valid inverse-relation index (relation 0's inverse = 0 + 46).
    """
    R_meta  = meta["w_R_unclamped"].size(0)
    R_model = model.mlp_feature.shape[0]
    if R_model != R_meta:
        raise ValueError(
            f"Rule count mismatch: model has {R_model}, rule_meta.pt has {R_meta}")

    # Build id_to_body from relation2rules (mirrors dump_rule_counts.py lines 371-374)
    id_to_body = {}
    for r_idx in range(model.num_relations * 2):
        for global_id, (r_head, r_body) in model.relation2rules[r_idx]:
            id_to_body[global_id] = (r_head, list(r_body))

    meta_head   = meta["head"].tolist()
    meta_length = meta["length"].tolist()
    mismatches  = 0

    for i in range(R_model):
        gid = model.rule_features[i, 0].item()
        if gid not in id_to_body:
            mismatches += 1
            if mismatches <= 3:
                print(f"  MISMATCH at row {i}: gid={gid} not in id_to_body")
            continue
        r_head, body = id_to_body[gid]
        if gid != i or r_head != meta_head[i] or len(body) != meta_length[i]:
            mismatches += 1
            if mismatches <= 3:
                print(f"  MISMATCH at row {i}: gid={gid} (exp {i}), "
                      f"head={r_head} (exp {meta_head[i]}), "
                      f"len={len(body)} (exp {meta_length[i]})")

    if mismatches:
        raise ValueError(f"Alignment failed: {mismatches}/{R_model} mismatches")
    print(f"  Alignment OK: {R_model} rules verified")


# ---------------------------------------------------------------------------
# Per-query MLP scoring
# ---------------------------------------------------------------------------

def score_queries_mlp(model: RulE, counts: dict,
                      hr2ooo: dict, N: int) -> list:
    """Score all queries with the MLP head; return list of (h, r, gold, L, H).

    Implements forward() scoring without calling forward() directly:
      1. For each query, reconstruct rule_count [R_fired, C_cand] from CSR.
      2. Pass through rule_to_entity -> score_model -> scatter -> + bias.
      3. Compute filtered rank (L, H) with same hr2ooo filter as original eval.
    """
    query_h    = counts["query_h"]
    query_r    = counts["query_r"]
    query_gold = counts["query_gold"]
    q_ptr      = counts["q_ptr"]
    rule_id    = counts["rule_id"]
    ent_arr    = counts["ent"]
    cnt_arr    = counts["cnt"].long()
    Q          = query_h.size(0)

    bias             = model.bias.detach().float()       # [N]
    rules_weight_emb = model.rules_weight_emb.detach()   # [R, 2000]
    mlp_feature      = model.mlp_feature.detach()        # [R, 100]

    def enc(h, r):
        return r * N + h

    results = []
    t0 = time.time()
    for qi in range(Q):
        if qi > 0 and qi % 500 == 0:
            elapsed = time.time() - t0
            print(f"    {qi}/{Q}  ({elapsed:.1f}s)")

        h    = query_h[qi].item()
        r    = query_r[qi].item()
        gold = query_gold[qi].item()
        s    = q_ptr[qi].item()
        e    = q_ptr[qi + 1].item()

        if s < e:
            r_ids = rule_id[s:e].tolist()
            ents  = ent_arr[s:e].tolist()
            cnts  = cnt_arr[s:e].tolist()

            unique_ents  = sorted(set(ents))
            unique_rules = sorted(set(r_ids))
            ent_to_cidx  = {ev: i for i, ev in enumerate(unique_ents)}
            rule_to_ridx = {rv: i for i, rv in enumerate(unique_rules)}

            C       = len(unique_ents)
            R_fired = len(unique_rules)

            # Reconstruct rule_count [R_fired, C] — mirrors forward() reshape
            # after torch.stack(rule_count_list)[:, candidate_set]
            rule_count = torch.zeros(R_fired, C, dtype=torch.float32)
            for rid, ei, cv in zip(r_ids, ents, cnts):
                rule_count[rule_to_ridx[rid], ent_to_cidx[ei]] = float(cv)

            fired_idx = torch.tensor(unique_rules, dtype=torch.long)
            r_emb  = rules_weight_emb[fired_idx]   # [R_fired, 2000]
            m_feat = mlp_feature[fired_idx]         # [R_fired, 100]

            # Forward through MLP head (mirrors forward lines 562-566)
            feat_out = model.rule_to_entity(rule_count, r_emb, m_feat)  # [C, 100]
            out      = model.score_model(feat_out).squeeze(-1)           # [C]

            # Scatter output into zeros, then add bias for ALL entities.
            # Mirrors forward lines 569-572:
            #   score = zeros; scatter(candidate_set, output); score += bias
            score = torch.zeros(N, dtype=torch.float32)
            cand_idx = torch.tensor(unique_ents, dtype=torch.long)
            score.scatter_(0, cand_idx, out.detach())
            score = score + bias
        else:
            # No rules fired: score = bias for all entities
            # (mirrors forward line 535: return mask + bias, where mask=zeros)
            score = bias.clone()

        # Filtered rank: competitors = entities NOT in hr2ooo[enc(h,r)]
        known_true = hr2ooo.get(enc(h, r), set())
        flag = torch.ones(N, dtype=torch.bool)
        for t in known_true:
            flag[t] = False

        val  = score[gold].item()
        n_gt = int((score[flag] > val).sum().item())
        n_ge = int((score[flag] >= val).sum().item())
        L    = n_gt + 1
        H    = n_ge + 2   # matches trainer.evaluate

        results.append((h, r, gold, L, H))

    elapsed = time.time() - t0
    print(f"    {Q}/{Q}  done in {elapsed:.1f}s")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# run.log validation references (rule-only MRR)
REFERENCES = {
    "valid": 0.806168,
    "test":  0.805089,
}
MRR_TOL = 2e-3


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--analysis_dir",
                   default="outputs/additive_umls_aggregation_2x2/analysis")
    p.add_argument("--checkpoint",
                   default="outputs/umls/grounding.pt")
    p.add_argument("--data_path",  default="data/umls")
    p.add_argument("--rule_file",  default="data/umls/mined_rules.txt")
    p.add_argument("--splits",     nargs="+", default=["valid", "test"])
    return p.parse_args()


def main():
    args = parse_args()

    _here = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.normpath(os.path.join(_here, ".."))
    os.chdir(_root)

    device = torch.device("cpu")

    N, num_rel = _load_entity_rel_sizes(args.data_path)
    print(f"N={N}  num_rel={num_rel}")

    # ------------------------------------------------------------------
    # Build model once; reuse across splits.
    # ------------------------------------------------------------------
    print("\nBuilding model and loading checkpoint ...")
    model, _ = build_and_load_model(
        args.checkpoint, args.rule_file, N, num_rel, device)

    print("\nChecking alignment with rule_meta.pt ...")
    meta = torch.load(
        os.path.join(args.analysis_dir, "rule_meta.pt"), weights_only=False)
    alignment_check(model, meta)

    print("\nBuilding hr2ooo filter ...")
    hr2ooo = build_hr2ooo(args.data_path, N, num_rel)

    # ------------------------------------------------------------------
    # Score each split.
    # ------------------------------------------------------------------
    for split in args.splits:
        print(f"\n=== Scoring split '{split}' ===")
        counts_path = os.path.join(args.analysis_dir, f"counts_{split}.pt")
        counts = torch.load(counts_path, weights_only=False)
        Q   = counts["query_h"].size(0)
        nnz = counts["rule_id"].size(0)
        print(f"  queries={Q}  nnz={nnz}")

        with torch.no_grad():
            results = score_queries_mlp(model, counts, hr2ooo, N)

        # ------------------------------------------------------------------
        # Metrics + validation
        # ------------------------------------------------------------------
        m = expectation_metrics(results)
        ref = REFERENCES.get(split)
        print(f"\n  Results:")
        hdr = f"  {'variant':<20}{'MRR':>9}{'Hit@1':>9}{'Hit@3':>9}{'Hit@10':>9}{'MR':>8}"
        print(hdr)
        print(f"  {'-'*(len(hdr)-2)}")
        print(f"  {'mlp_original':<20}{m['MRR']:>9.6f}{m['Hit@1']:>9.6f}"
              f"{m['Hit@3']:>9.6f}{m['Hit@10']:>9.6f}{m['MR']:>8.2f}")
        if ref is not None:
            delta  = abs(m["MRR"] - ref)
            status = "PASS" if delta < MRR_TOL else "FAIL"
            print(f"\n  MRR validation vs run.log: "
                  f"got={m['MRR']:.6f}  ref={ref:.6f}  "
                  f"delta={delta:.2e}  [{status}]")

        # ------------------------------------------------------------------
        # Write ranks_mlp_<split>.csv
        # ------------------------------------------------------------------
        ranks_csv = os.path.join(args.analysis_dir, f"ranks_mlp_{split}.csv")
        with open(ranks_csv, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["h", "r", "gold", "L_mlp", "H_mlp"])
            for h, r, gold, L, H in results:
                writer.writerow([h, r, gold, L, H])
        print(f"  Wrote {ranks_csv}  ({Q} rows)")

    print("\nDone.")


if __name__ == "__main__":
    main()
