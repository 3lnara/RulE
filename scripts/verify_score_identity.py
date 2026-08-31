#!/usr/bin/env python3
"""Assert the attribution identity `score_model(t) == sum_R beta_R * 1[rule R
fired h -> t]` that the interpretability claim rests on.

Runs the real RulE.forward on the real KnowledgeGraph under
the --logreg_binary flags (conf_count, conf_binary, use_bias=False) and compares
the per-entity score against two independent reconstructions -- from the dumped
counts CSR and from a fresh MinimalGraph grounding. beta is synthetic (dense,
signed) by default as a stronger probe of the rule_id -> weight map; --logreg_file
uses a fitted beta.

Usage:
    python scripts/verify_score_identity.py --data_path data/<ds> \\
        --counts outputs/<ds>/rq/counts_valid.pt
"""

import argparse
import json
import os
import sys

import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, ".."))
_SRC_DIR = os.path.join(_REPO_ROOT, "src_additive")
for _p in (_SRC_DIR, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import data as _data_mod
import model as _model_mod
import dump_rule_counts as _drc

KnowledgeGraph = _data_mod.KnowledgeGraph
RulE = _model_mod.RulE
MinimalGraph = _drc.MinimalGraph
load_rules = _drc.load_rules


class Report:
    def __init__(self):
        self.failures = []
        self.checks = 0

    def check(self, cond, msg, detail=""):
        self.checks += 1
        if cond:
            print(f"    PASS  {msg}")
        else:
            self.failures.append(msg)
            print(f"    FAIL  {msg}")
            if detail:
                for ln in str(detail).splitlines():
                    print(f"          {ln}")
        return cond


def build_model(graph, rules, beta, device, cfg=None):
    """RulE under exactly the flags trainer.py sets for --logreg_binary.

    Architecture dims come from the run config (same keys dump_rule_counts.py
    reads) so this builds on any dataset, not just UMLS. Under these flags the
    entity/relation embeddings never enter the score, so the dims only affect
    instantiation -- the checked identity is invariant to them.
    """
    cfg = cfg or {}
    model = RulE(graph=graph,
                 p_norm=int(cfg.get("p_norm", 2)),
                 mlp_rule_dim=int(cfg.get("mlp_rule_dim", 100)),
                 gamma_fact=float(cfg.get("gamma_fact", 6)),
                 gamma_rule=float(cfg.get("gamma_rule", 8)),
                 hidden_dim=int(cfg.get("hidden_dim", 2000)),
                 device=device,
                 dataset=cfg.get("data_path", "umls"))
    model.set_rules(rules)

    # Mirror trainer.py:437-448 verbatim -- including the beta[gids] remap.
    gids = model.rule_features[:, 0].long()
    w = torch.nan_to_num(beta[gids], nan=0.0)
    model.register_buffer("rule_weight_logit", w.to(device).clone())
    model.conf_count = True
    model.conf_binary = True
    model.use_bias = False
    model.bias.requires_grad_(False)
    model.eval()
    return model


def csr_rows(counts, q):
    """(rule_id, ent, cnt) slice for query q from the CSR dump."""
    lo, hi = int(counts["q_ptr"][q]), int(counts["q_ptr"][q + 1])
    return (counts["rule_id"][lo:hi].long(),
            counts["ent"][lo:hi].long(),
            counts["cnt"][lo:hi].long())


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_path", default="data/umls")
    p.add_argument("--rule_file", default=None)
    p.add_argument("--config", default=None,
                   help="Run config json (umls_config.json etc.) for model dims, so "
                        "this builds on any dataset. Under --logreg_binary the "
                        "dims do not affect the checked score.")
    p.add_argument("--counts",
                   default="outputs/additive_umls_aggregation_2x2/analysis/counts_valid.pt")
    p.add_argument("--logreg_file", default=None,
                   help="Use a fitted beta from rule_logreg.pt instead of synthetic.")
    p.add_argument("--num_queries", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tol", type=float, default=1e-4,
                   help="Abs tolerance for the float32 score comparison. The sums "
                        "are over the same terms in a different order, so only "
                        "float non-associativity separates them.")
    args = p.parse_args()

    os.chdir(_REPO_ROOT)
    device = torch.device("cpu")
    torch.manual_seed(args.seed)
    rule_file = args.rule_file or os.path.join(args.data_path, "mined_rules.txt")
    cfg = {}
    if args.config:
        with open(args.config) as fh:
            cfg = json.load(fh)

    print(f"data_path : {args.data_path}")
    print(f"counts    : {args.counts}")

    kg = KnowledgeGraph(args.data_path)
    mg = MinimalGraph(args.data_path)
    rules = load_rules(rule_file, mg.relation_size)
    R = len(rules)
    N = mg.entity_size

    if args.logreg_file:
        beta = torch.load(args.logreg_file, map_location="cpu")["beta"].float()
        src = f"fitted ({args.logreg_file})"
    else:
        # Dense + signed + no ties: a permutation or off-by-one in the
        # rule_id -> beta map cannot survive this.
        beta = torch.randn(R, dtype=torch.float32)
        src = "synthetic randn (dense, signed)"
    print(f"beta      : {src}  R={R}  "
          f"neg={int((beta < 0).sum())}  zero={int((beta == 0).sum())}")

    counts = torch.load(args.counts, map_location="cpu")
    Q = counts["query_h"].numel()

    model = build_model(kg, rules, beta, device, cfg)
    rep = Report()

    # Group sampled queries by relation: forward() asserts a single query_r.
    g = torch.Generator().manual_seed(args.seed)
    sel = torch.randperm(Q, generator=g)[:args.num_queries].tolist()
    by_rel = {}
    for q in sel:
        by_rel.setdefault(int(counts["query_r"][q]), []).append(q)

    print(f"\n[1] score identity on {len(sel)} valid queries "
          f"across {len(by_rel)} relations")

    max_d_a = max_d_b = 0.0
    n_nonzero = 0
    inversions = 0
    gold_rank_bad = []
    n_tied = 0
    attrib_bad = []

    for r, qs in sorted(by_rel.items()):
        H = torch.tensor([int(counts["query_h"][q]) for q in qs], dtype=torch.long)
        Rr = torch.full((len(qs),), r, dtype=torch.long)

        with torch.no_grad():
            score, _ = model(H, Rr, None)              # [B, N] REAL forward
        score = score.float()

        # ---- REF-A: reconstruct from the dumped CSR ------------------------
        ref_a = torch.zeros(len(qs), N, dtype=torch.float64)
        for i, q in enumerate(qs):
            rid, ent, cnt = csr_rows(counts, q)
            fired = cnt > 0
            ref_a[i].index_add_(0, ent[fired], beta[rid[fired]].double())

        # ---- REF-B: reconstruct from a fresh MinimalGraph grounding --------
        ref_b = torch.zeros(len(qs), N, dtype=torch.float64)
        for gid, r_head, body in [(x[0], x[1], x[2:]) for x in rules if x[1] == r]:
            fired = (mg.grounding(H, r_head, body, None) > 0).double()   # [B, N]
            ref_b += beta[gid].double() * fired

        n_nonzero += int((score.abs() > 0).sum())
        max_d_a = max(max_d_a, float((score.double() - ref_a).abs().max()))
        max_d_b = max(max_d_b, float((score.double() - ref_b).abs().max()))

        # Ranking agreement: binary scores tie heavily, so check no strictly
        # ordered pair is inverted (reference scores in the model's order must be
        # non-increasing up to tol), not an identical permutation.
        for i in range(len(qs)):
            order = score[i].argsort(descending=True)
            walk = ref_a[i][order]
            inversions += int((walk[:-1] - walk[1:] < -args.tol).sum())
            n_tied += int(len(walk) - torch.unique(ref_a[i]).numel())

            # Gold rank under a fixed tie-break, computed from each side
            # independently: optimistic (# strictly better) and pessimistic
            # (# better or equal). Both must agree.
            gold = int(counts["query_gold"][qs[i]])
            for name, s in (("model", score[i].double()), ("ref", ref_a[i])):
                opt = int((s > s[gold] + args.tol).sum()) + 1
                pes = int((s >= s[gold] - args.tol).sum())
                if name == "model":
                    m_opt, m_pes = opt, pes
                else:
                    if (opt, pes) != (m_opt, m_pes):
                        gold_rank_bad.append((qs[i], (m_opt, m_pes), (opt, pes)))

        # ---- attribution completeness on the gold tail ---------------------
        for i, q in enumerate(qs):
            gold = int(counts["query_gold"][q])
            rid, ent, cnt = csr_rows(counts, q)
            m = (ent == gold) & (cnt > 0)
            contrib = beta[rid[m]].double().sum()
            if abs(float(score[i, gold]) - float(contrib)) > args.tol:
                attrib_bad.append((q, float(score[i, gold]), float(contrib)))

    rep.check(n_nonzero > 0, f"scores are non-trivial ({n_nonzero} nonzero cells)",
              "all scores zero; the identity is vacuous")
    rep.check(max_d_a <= args.tol,
              f"model score == beta . 1[cnt>0] from counts_valid.pt "
              f"(max |diff| = {max_d_a:.2e})")
    rep.check(max_d_b <= args.tol,
              f"model score == beta . 1[fired] from fresh grounding "
              f"(max |diff| = {max_d_b:.2e})")
    rep.check(inversions == 0,
              f"no strictly-ordered pair inverted ({len(sel)} queries; "
              f"{n_tied} tied cells, ordering agrees up to ties)",
              f"{inversions} adjacent inversions beyond tol={args.tol:g}")
    rep.check(not gold_rank_bad,
              "gold rank (optimistic and pessimistic) identical model vs reference",
              f"first mismatches (q, model(opt,pes), ref(opt,pes)): {gold_rank_bad[:3]}")
    rep.check(not attrib_bad,
              "gold score == sum of its firing rules' beta (attribution complete)",
              f"first mismatches (q, score, sum_contrib): {attrib_bad[:3]}")

    # ---- [2] intercept rank-invariance --------------------------------------
    print("\n[2] intercept omission is rank-invariant")
    r0 = sorted(by_rel)[0]
    qs = by_rel[r0]
    H = torch.tensor([int(counts["query_h"][q]) for q in qs], dtype=torch.long)
    with torch.no_grad():
        s0, _ = model(H, torch.full((len(qs),), r0, dtype=torch.long), None)
    b = 3.7159
    same = all(torch.equal(s0[i].argsort(descending=True),
                           (s0[i] + b).argsort(descending=True))
               for i in range(len(qs)))
    rep.check(same, f"adding a constant intercept b={b} leaves every ranking fixed")

    print("\n" + "=" * 74)
    if rep.failures:
        print(f"FAILED  {len(rep.failures)}/{rep.checks} checks:")
        for f in rep.failures:
            print(f"  - {f}")
        print("=" * 74)
        return 1
    print(f"ALL {rep.checks} CHECKS PASSED -- score is exactly the sum of per-rule beta.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
