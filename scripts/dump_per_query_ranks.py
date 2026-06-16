#!/usr/bin/env python3
"""
Dump per-query ranking data for a single RulE checkpoint.

Produces a CSV with columns:
  h, r, t, grounded, L_rule, H_rule, L_kge, H_kge, L_fused, H_fused

where (L, H) brackets encode the filtered rank range (fractional / tie-broken
ranking uses both endpoints). Rank = (L + H - 1) / 2 is the standard mid-point.

  rule-only  : only the rule score; ungrounded target -> L=1, H=num_entities+1
  kge-only   : only the KGE score; all entities are always scored
  fused      : rule_score + alpha * kge_score; same mask fallback as rule-only

Usage (additive backbone, existing baseline):
    python scripts/dump_per_query_ranks.py \\
        --backend additive \\
        --config  outputs/additive_umls_baseline/config.json \\
        --checkpoint outputs/additive_umls_baseline/grounding.pt \\
        --data_path data/umls \\
        --rule_file data/umls/mined_rules.txt \\
        --alpha 2.0 \\
        --out_csv outputs/additive_umls_baseline/ranks.csv

Usage (GAT backbone):
    python scripts/dump_per_query_ranks.py \\
        --backend gat \\
        --config  outputs/gat_umls_rule_attn/config.json \\
        --checkpoint outputs/gat_umls_rule_attn/grounding.pt \\
        --data_path data/umls \\
        --rule_file data/umls/mined_rules.txt \\
        --alpha 2.0 \\
        --out_csv outputs/gat_umls_rule_attn/ranks.csv
"""

import argparse
import csv
import importlib
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint(model: nn.Module, path: str, device: torch.device) -> None:
    ckpt = torch.load(path, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    elif "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)


def import_backend(src_root: str, backend: str):
    src_dir_map = {
        "additive": os.path.join(src_root, "src_additive"),
        "gat":      os.path.join(src_root, "src"),
    }
    src_dir = src_dir_map.get(backend)
    if src_dir is None:
        raise ValueError(f"Unknown backend '{backend}'. Choose 'additive' or 'gat'.")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    data_mod  = importlib.import_module("data")
    model_mod = importlib.import_module("model")
    utils_mod = importlib.import_module("utils")
    return data_mod, model_mod, utils_mod


# ---------------------------------------------------------------------------
# Model construction + checkpoint loading
# ---------------------------------------------------------------------------

def build_additive_model(cfg, graph, rules, device, data_path):
    """Reconstruct an additive-backbone model and register all buffers/params
    that the trainer adds dynamically, then load the checkpoint."""
    data_mod, model_mod, utils_mod = import_backend(
        os.path.join(os.path.dirname(__file__), ".."), "additive"
    )

    model = model_mod.RulE(
        graph=graph,
        p_norm=cfg.p_norm,
        mlp_rule_dim=cfg.mlp_rule_dim,
        gamma_fact=cfg.gamma_fact,
        gamma_rule=cfg.gamma_rule,
        hidden_dim=cfg.hidden_dim,
        device=device,
        dataset=data_path,
    )
    model.set_rules(rules)
    num_rules = model.rule_features.size(0)

    # Flags --------------------------------------------------------------
    model.simple_aggregation = True
    model.paper_sum = getattr(cfg, "paper_sum", False)
    model.learnable_rule_weight = getattr(cfg, "learnable_rule_weight", False)

    # rule_weight_logit: register as buffer (frozen baseline or paper_sum)
    # or as Parameter (Lever-1 learnable).  The checkpoint will overwrite
    # the values; we only need the right key type before load_state_dict.
    dummy_logits = torch.zeros(num_rules, device=device)
    if model.learnable_rule_weight:
        model.rule_weight_logit = nn.Parameter(dummy_logits)
    else:
        model.register_buffer("rule_weight_logit", dummy_logits)

    # FM embedding -------------------------------------------------------
    if getattr(cfg, "fm_interactions", False):
        k = int(getattr(cfg, "fm_rank", 16))
        model.rule_fm_emb = nn.Parameter(
            torch.zeros(num_rules, k, device=device)
        )
        model.use_fm = True
    else:
        model.use_fm = False

    # NAM ----------------------------------------------------------------
    if getattr(cfg, "nam", False):
        from layers import MLP  # noqa: F401 – imported for model_mod's MLP
        nam_dim    = int(getattr(cfg, "nam_dim", 16))
        nam_hidden = int(getattr(cfg, "nam_hidden", 64))
        model.nam_emb = nn.Parameter(
            torch.zeros(num_rules, nam_dim, device=device)
        )
        model.nam_net = model_mod.MLP(2 + nam_dim, [nam_hidden, 1]).to(device)
        model.use_nam = True
    else:
        model.use_nam = False

    return model


def build_gat_model(cfg, graph, rules, device, data_path):
    """Reconstruct a GAT-backbone model and replay the variant-a confidence
    setup that the GAT trainer does after eval_compute_rule_weight."""
    _, model_mod, _ = import_backend(
        os.path.join(os.path.dirname(__file__), ".."), "gat"
    )

    model = model_mod.RulE(
        graph=graph,
        p_norm=cfg.p_norm,
        mlp_rule_dim=cfg.mlp_rule_dim,
        gamma_fact=cfg.gamma_fact,
        gamma_rule=cfg.gamma_rule,
        hidden_dim=cfg.hidden_dim,
        device=device,
        dataset=data_path,
        attn_dim=getattr(cfg, "attn_dim", 128),
        gat_variant=getattr(cfg, "gat_variant", "baseline"),
    )
    model.set_rules(rules)
    return model


# ---------------------------------------------------------------------------
# Ranking helpers
# ---------------------------------------------------------------------------

def lh_rank(scores_row, flag_row, true_t):
    """Return (L, H) filtered rank of true_t in scores_row.

    flag_row: 1D bool tensor -- True for entities that are NOT legitimate
              correct answers other than t (i.e., the standard filtered setting
              masks out other known positives).  True -> include in competition.
    """
    val = scores_row[true_t]
    L = int((scores_row[flag_row] > val).sum().item()) + 1
    H = int((scores_row[flag_row] >= val).sum().item()) + 2
    return L, H


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def dump_ranks(model, test_loader, device, alpha, num_entities, out_csv):
    model.eval()
    rows = []

    for batch in test_loader:
        all_h, all_r, all_t, flag = batch
        all_h = all_h.squeeze(0).to(device)
        all_r = all_r.squeeze(0).to(device)
        all_t = all_t.squeeze(0).to(device)
        flag  = flag.squeeze(0).to(device)          # [batch, entities] bool

        rule_score, mask = model(all_h, all_r, None) # [batch, entities]
        kge              = model.compute_g_KGE(all_h, all_r)
        fused            = rule_score + alpha * kge

        # `_last_ground_mask` is set by the additive forward as a side-effect;
        # for the GAT backend it is not set, so fall back to `mask`.
        ground_mask = getattr(model, "_last_ground_mask", mask)

        for i in range(all_t.size(0)):
            t     = all_t[i].item()
            h     = all_h[i].item()
            r     = all_r[i].item()
            fi    = flag[i]           # [entities] bool filter

            grounded = bool(ground_mask[i, t].item())

            # rule-only
            if mask[i, t].item():
                L_rule, H_rule = lh_rank(rule_score[i], fi, t)
            else:
                L_rule, H_rule = 1, num_entities + 1

            # kge-only (all entities always scored)
            L_kge, H_kge = lh_rank(kge[i], fi, t)

            # fused (same mask fallback as rule-only)
            if mask[i, t].item():
                L_fused, H_fused = lh_rank(fused[i], fi, t)
            else:
                L_fused, H_fused = 1, num_entities + 1

            rows.append({
                "h": h, "r": r, "t": t, "grounded": int(grounded),
                "L_rule": L_rule, "H_rule": H_rule,
                "L_kge":  L_kge,  "H_kge":  H_kge,
                "L_fused": L_fused, "H_fused": H_fused,
            })

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    fieldnames = ["h", "r", "t", "grounded",
                  "L_rule", "H_rule", "L_kge", "H_kge", "L_fused", "H_fused"]
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_csv}")
    return rows


# ---------------------------------------------------------------------------
# Quick aggregate metrics (for sanity)
# ---------------------------------------------------------------------------

def compute_metrics(rows, key_l, key_h):
    hit1 = hit3 = hit10 = mr = mrr = 0.0
    n = 0
    for row in rows:
        L, H = row[key_l], row[key_h]
        for rank in range(L, H):
            w = 1.0 / (H - L)
            if rank <= 1:  hit1 += w
            if rank <= 3:  hit3 += w
            if rank <= 10: hit10 += w
            mr  += w * rank
            mrr += w / rank
        n += 1
    if n == 0:
        return {}
    return {
        "MRR":   mrr  / n,
        "MR":    mr   / n,
        "Hit@1": hit1 / n,
        "Hit@3": hit3 / n,
        "Hit@10": hit10 / n,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Dump per-query filtered ranks for a RulE checkpoint.")
    p.add_argument("--backend", required=True, choices=["additive", "gat"])
    p.add_argument("--config",  required=True,
                   help="Path to the checkpoint's config.json (sibling of grounding.pt).")
    p.add_argument("--checkpoint", required=True,
                   help="Path to grounding.pt.")
    p.add_argument("--data_path", required=True,
                   help="Local path to the dataset directory (e.g. data/umls).")
    p.add_argument("--rule_file", required=True,
                   help="Local path to the rule file (e.g. data/umls/mined_rules.txt).")
    p.add_argument("--alpha", type=float, default=None,
                   help="KGE fusion weight for the fused score. Defaults to cfg.alpha.")
    p.add_argument("--out_csv", required=True,
                   help="Output CSV path.")
    p.add_argument("--src_root", default=None,
                   help="Root of the RulE repo (parent of src_additive/ and src/). "
                        "Defaults to the parent of the scripts/ directory.")
    p.add_argument("--seed", type=int, default=800)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # Resolve src_root so we can add the right backend src dir to sys.path
    if args.src_root is None:
        args.src_root = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..")
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load config ---------------------------------------------------------
    # Import utils from the relevant backend to reuse load_config.
    data_mod, model_mod, utils_mod = import_backend(args.src_root, args.backend)

    configs = utils_mod.load_config(args.config)
    cfg = configs[0] if isinstance(configs, (list, tuple)) else configs

    alpha = args.alpha if args.alpha is not None else float(getattr(cfg, "alpha", 2.0))
    print(f"alpha = {alpha}")

    # Build graph + rules -------------------------------------------------
    graph   = data_mod.KnowledgeGraph(args.data_path)
    ruleset = data_mod.RuleDataset(
        graph.relation_size, args.rule_file,
        getattr(cfg, "rule_negative_size", 64),
    )
    rules = [rule[0] for rule in ruleset.rules]

    # Build model ---------------------------------------------------------
    if args.backend == "additive":
        model = build_additive_model(cfg, graph, rules, device, args.data_path)
        load_checkpoint(model, args.checkpoint, device)
    else:  # gat
        model = build_gat_model(cfg, graph, rules, device, args.data_path)
        load_checkpoint(model, args.checkpoint, device)
        model.eval_compute_rule_weight(device)
        if getattr(cfg, "use_rule_confidence_variant_a", False):
            rule_features = model.rule_features.to(device)
            rule_masks    = model.rule_masks.to(device)
            split_num = max(1, rule_features.size(0) // 128)
            scalars = []
            with torch.no_grad():
                for rf, rm in zip(
                    torch.split(rule_features, split_num, 0),
                    torch.split(rule_masks,    split_num, 0),
                ):
                    s, _ = model.add_ruleE(rf.unsqueeze(1), rm)
                    scalars.append(s.squeeze(1))
            scalar_conf = torch.sigmoid(torch.cat(scalars, dim=0))
            model.grounding_gat.set_frozen_confidences(scalar_conf)
            print("GAT variant_a: frozen confidences rebuilt from rule embeddings.")

    model = model.to(device)
    model.eval()

    # Build test loader ---------------------------------------------------
    batch_size = getattr(cfg, "g_batch_size", 16)
    from torch.utils.data import DataLoader
    test_loader = DataLoader(
        data_mod.TestDataset(graph, batch_size),
        batch_size=1, num_workers=0,
    )

    # Dump ----------------------------------------------------------------
    rows = dump_ranks(
        model, test_loader, device, alpha,
        num_entities=graph.entity_size,
        out_csv=args.out_csv,
    )

    # Sanity metrics ------------------------------------------------------
    for label, lk, hk in [("rule-only", "L_rule", "H_rule"),
                           ("kge-only",  "L_kge",  "H_kge"),
                           ("fused",     "L_fused", "H_fused")]:
        m = compute_metrics(rows, lk, hk)
        print(f"{label:12s}  MRR={m['MRR']:.4f}  Hit@10={m['Hit@10']:.4f}  MR={m['MR']:.1f}")


if __name__ == "__main__":
    main()
