#!/usr/bin/env python3
"""
Fit an interpretable per-rule LOGISTIC-REGRESSION aggregator for RulE.

For every rule R (head relation r, body path B) we learn ONE weight beta_R,
plus ONE shared intercept b, by binary cross-entropy on a leave-one-out (LOO)
train design matrix. This is the learned counterpart of the frozen
precision_binary baseline: same in-model score form

    score(t) = sum_R beta_R * 1[rule R fired h->t]   (+ b, constant per query)

but with beta_R fitted to the data instead of set to the rule's empirical PCA
precision (or the frozen RulE embedding confidence).

--------------------------------------------------------------------------------
Design matrix (leave-one-out)
--------------------------------------------------------------------------------
Rows are candidate (h, t) pairs, one row per pair for which >=1 rule of the head
relation r fires. Columns are rules. Entries are binary firing indicators:

    x[(h,t), R] = 1[body of R grounds h -> t]

Label y[(h,t)] = 1 iff t is a true train tail of (h, r), else 0.

LEAVE-ONE-OUT (edge-specific): a positive (h, t) is grounded with exactly the
one answer edge (h, r, t) removed -- the same edges_to_remove semantics
src_additive/data.py uses during training -- so the trivial "length-1 rule =
the head relation" cannot reuse its own answer, while legitimate multi-hop paths
that merely start with a DIFFERENT r-edge out of h (e.g. transitivity
r(x,y)&r(y,z)->r(x,z)) are preserved. Negatives use plain full-graph grounding
(their own edge does not exist, so there is nothing to leave out). This is the
intended difference from the precision baseline, which grounds on the full train
graph with no leave-one-out.

Only fired (h,t) pairs are rows: entities no body path can reach are never
scored at eval time, so they carry no signal and are excluded, exactly matching
the in-model behaviour. A gold pair reachable only via its own edge therefore
drops out under leave-one-out, exactly as it is unrankable at eval time.

--------------------------------------------------------------------------------
Fit
--------------------------------------------------------------------------------
    logits = X @ beta + b
    loss   = data_loss(logits, y) [+ optional pos_weight]
             + l2 * ||beta||^2

data_loss is one of BCE (logistic regression), MSE (OLS), or squared hinge
(linear SVM), selected by --loss; see fit_logreg's docstring. All three share the
identical in-model score form, so nothing downstream changes: the fitted beta is
swept and selected by scripts/select_logreg.py exactly the same way, and
--logreg_binary loads it unchanged. Rules that never fire get beta_R = 0 (no rows
reference them), and contribute nothing at eval -- matching precision_binary's
nan->0 handling.

The intercept b is learned (it matters for the BCE fit / calibration) but is a
CONSTANT added to every candidate tail within a query, so it does NOT change the
per-query ranking. In-model evaluation therefore matches precision_binary
exactly (binary activation, no per-entity bias, per-rule weights); the aggregator
just supplies learned beta_R in place of frozen precision.

--------------------------------------------------------------------------------
Outputs (all written to --out_dir)
--------------------------------------------------------------------------------
  rule_logreg.pt         tensors indexed by rule_id:
                           beta, support, gold_fired, head, length
                         plus scalar `intercept` and a `meta` dict.
  rule_logreg_train.csv  per-rule table (one row per rule)

Alignment note: rule_id == position in load_rules() output == rule_features[:, 0]
in the model, so beta[rule_id] lines up with the model's rule rows exactly the
way rule_precision.pt's `precision` does. Load it in-model via
    src_additive/main.py --logreg_binary --logreg_file <out_dir>/rule_logreg.pt

Usage (from repo root):
    python scripts/rule_logreg_train.py \\
        --data_path data/umls \\
        --rule_file data/umls/mined_rules.txt \\
        --out_dir   outputs/additive_umls_aggregation_2x2/logreg \\
        --device    cuda
"""

import argparse
import csv
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Bootstrap: reuse MinimalGraph and load_rules from dump_rule_counts.py.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.normpath(os.path.join(_SCRIPT_DIR, ".."))

sys.path.insert(0, _SCRIPT_DIR)
import dump_rule_counts as _drc
MinimalGraph = _drc.MinimalGraph
load_rules   = _drc.load_rules


# ---------------------------------------------------------------------------
# Build gold dict: gold[r][h] = set of true train tails for (h, r)
# ---------------------------------------------------------------------------

def build_gold(graph: MinimalGraph) -> dict:
    """gold[r][h] = set of true train tails for directed relation r, head h.

    graph.train_facts already includes both forward (h, r, t) and inverse
    (t, r+NR, h) triples, so this covers all directed relations uniformly.
    """
    gold: dict = defaultdict(lambda: defaultdict(set))
    for h, r, t in graph.train_facts:
        gold[r][h].add(t)
    return gold


# ---------------------------------------------------------------------------
# Grounding (batched over query heads), optionally with edge-specific
# leave-one-out on the query relation.
# ---------------------------------------------------------------------------

def ground(
    graph: MinimalGraph,
    heads: torch.Tensor,            # [B] head entity ids, on `device`
    r_body: list,
    device: torch.device,
    r_query: int = None,            # query relation (only used with drop_dst)
    drop_dst: torch.Tensor = None,  # [B] held-out answer tail per column, or None
) -> torch.Tensor:
    """Ground rule body `r_body` from each head. Returns [B, N] int64 counts.

    If `drop_dst` is given, apply EDGE-SPECIFIC leave-one-out: at every hop that
    traverses the query relation r_query, only the single edge
    (heads[b], r_query, drop_dst[b]) is zeroed for column b. This removes just
    that column's held-out answer edge -- the exact semantics of
    src_additive/data.py's grounding (edges_to_remove) during training -- so a
    body path that merely begins by taking a DIFFERENT r_query edge out of the
    head (e.g. transitivity r(x,y)&r(y,z)->r(x,z)) is preserved.

    With drop_dst=None this is a plain full-graph grounding (no removal),
    matching MinimalGraph.grounding.
    """
    N = graph.entity_size
    B = heads.size(0)
    x = torch.zeros(N, B, dtype=torch.long, device=device)
    x.scatter_(0, heads.unsqueeze(0), 1)               # one-hot init [N, B]

    do_loo = drop_dst is not None
    for rb in r_body:
        src, dst = graph.relation2adjacency[rb]
        if src.numel() == 0:
            return torch.zeros(B, N, dtype=torch.long, device=device)
        if src.device != device:
            src = src.to(device)
            dst = dst.to(device)
            graph.relation2adjacency[rb] = (src, dst)

        messages = x[src]                              # [E, B]
        if do_loo and rb == r_query:
            # Zero the message on edge (heads[b], r_query, drop_dst[b]) for
            # column b: match BOTH endpoints so only the held-out answer edge
            # is dropped, not every r_query edge out of the head.
            drop = ((src.unsqueeze(1) == heads.unsqueeze(0)) &
                    (dst.unsqueeze(1) == drop_dst.unsqueeze(0)))    # [E, B]
            messages = messages.masked_fill(drop, 0)

        new_x = torch.zeros(N, B, dtype=x.dtype, device=device)
        new_x.scatter_add_(0, dst.unsqueeze(1).expand_as(messages), messages)
        x = new_x

    return x.transpose(0, 1)                           # [B, N]


# ---------------------------------------------------------------------------
# Design matrix construction (two passes per relation)
# ---------------------------------------------------------------------------

def build_design_matrix(
    graph: MinimalGraph,
    rules: list,
    gold: dict,
    device: torch.device,
    chunk_size: int = 0,
):
    """Build the leave-one-out binary design matrix.

    Rows are candidate (h, t) pairs; columns are rules; entries are binary rule
    firings x[(h,t), R] = 1[body of R grounds h->t]. Label y = 1[t is a true
    train tail of (h, r)]. Built in two passes per head relation r:

      PASS A (negatives): plain full-graph grounding from each known head. Every
        fired (h, t') pair with t' NOT a gold tail is a negative row (label 0).
        Deduplicated by head, so each (h, t') appears once.

      PASS B (positives): for each gold fact (h, r, t), ground with EDGE-SPECIFIC
        leave-one-out removing exactly (h, r, t). If >=1 rule still fires h->t,
        (h, t) is a positive row (label 1) with that leave-one-out firing. Gold
        pairs no rule can reach without their own edge are dropped (they are
        unrankable at eval time too), so the trivial identity rule (body = the
        head relation) correctly contributes nothing.

    Returns rows, cols, y, fired_total, gold_fired, M (see fit_logreg).
    """
    R = len(rules)
    fired_total = np.zeros(R, dtype=np.int64)   # rows (pos+neg) each rule fires in
    gold_fired  = np.zeros(R, dtype=np.int64)   # positive rows each rule fires in

    relation2rules: dict = defaultdict(list)
    for rule in rules:
        gid    = rule[0]
        r_head = rule[1]
        body   = rule[2:]
        relation2rules[r_head].append((gid, r_head, body))

    N = graph.entity_size

    rows_chunks, cols_chunks, y_chunks = [], [], []
    row_offset = 0

    def _emit_rows(fired_list, keep_mask, bt_index, labels):
        """Append COO entries for the rows selected by keep_mask.

        fired_list : list of (gid, fired[Bk, N] bool)
        keep_mask  : [Bk, N] bool  -- which (i_head, t) cells become rows
        bt_index   : [M_i, 2] nonzero indices of keep_mask (i_head, t)
        labels     : [M_i] float row labels
        """
        nonlocal row_offset
        M_i = bt_index.size(0)
        if M_i == 0:
            return
        row_map = torch.full((keep_mask.size(0), N), -1,
                             dtype=torch.long, device=device)
        row_map[bt_index[:, 0], bt_index[:, 1]] = torch.arange(M_i, device=device)
        y_chunks.append(labels.cpu())
        for gid, fired in fired_list:
            sel = fired & keep_mask
            fbt = sel.nonzero(as_tuple=False)
            if fbt.size(0) == 0:
                continue
            r_ids = row_map[fbt[:, 0], fbt[:, 1]] + row_offset
            rows_chunks.append(r_ids.cpu())
            cols_chunks.append(torch.full((fbt.size(0),), gid, dtype=torch.long))
            fired_total[gid] += fbt.size(0)
        row_offset += M_i

    t0 = time.time()
    for r, rule_list in sorted(relation2rules.items()):
        known_h = sorted(gold[r].keys())
        if not known_h:
            continue

        # ---- PASS A: negatives (plain grounding, batched over heads) --------
        Bn = len(known_h)
        step = Bn if (chunk_size is None or chunk_size <= 0) else chunk_size
        for blk in range(0, Bn, step):
            block_h = known_h[blk:blk + step]
            Bk = len(block_h)
            H  = torch.tensor(block_h, dtype=torch.long, device=device)

            gold_mask = torch.zeros(Bk, N, dtype=torch.bool, device=device)
            for i, h in enumerate(block_h):
                for t in gold[r][h]:
                    gold_mask[i, t] = True

            fired_list = []
            neg_any = torch.zeros(Bk, N, dtype=torch.bool, device=device)
            for gid, r_head, body in rule_list:
                with torch.no_grad():
                    fired = ground(graph, H, body, device) > 0     # [Bk, N]
                fired_list.append((gid, fired))
                neg_any |= (fired & ~gold_mask)
            bt = neg_any.nonzero(as_tuple=False)
            labels = torch.zeros(bt.size(0), device=device)         # all negatives
            _emit_rows(fired_list, neg_any, bt, labels)

        # ---- PASS B: positives (edge-specific LOO, batched over gold facts) --
        facts = [(h, t) for h in known_h for t in sorted(gold[r][h])]
        Bf = len(facts)
        for blk in range(0, Bf, step):
            block = facts[blk:blk + step]
            Bk = len(block)
            H = torch.tensor([h for h, _ in block], dtype=torch.long, device=device)
            T = torch.tensor([t for _, t in block], dtype=torch.long, device=device)

            # answer_cell[b, t] True only at (b, T[b]) -- the held-out positive.
            answer_cell = torch.zeros(Bk, N, dtype=torch.bool, device=device)
            answer_cell[torch.arange(Bk, device=device), T] = True

            fired_list = []
            pos_any = torch.zeros(Bk, N, dtype=torch.bool, device=device)
            for gid, r_head, body in rule_list:
                with torch.no_grad():
                    fired = ground(graph, H, body, device,
                                   r_query=r, drop_dst=T) > 0        # [Bk, N]
                fired_at_answer = fired & answer_cell                # keep only col T[b]
                fired_list.append((gid, fired_at_answer))
                pos_any |= fired_at_answer
            bt = pos_any.nonzero(as_tuple=False)
            labels = torch.ones(bt.size(0), device=device)          # all positives
            # gold_fired = per-rule count over positive rows
            for gid, fired_at_answer in fired_list:
                gold_fired[gid] += int((fired_at_answer & pos_any).sum().item())
            _emit_rows(fired_list, pos_any, bt, labels)

    elapsed = time.time() - t0
    M = row_offset
    if rows_chunks:
        rows = torch.cat(rows_chunks)
        cols = torch.cat(cols_chunks)
        y    = torch.cat(y_chunks)
    else:
        rows = torch.zeros(0, dtype=torch.long)
        cols = torch.zeros(0, dtype=torch.long)
        y    = torch.zeros(0, dtype=torch.float)

    print(f"  Design matrix built in {elapsed:.1f}s  "
          f"rows(M)={M}  nnz={rows.numel()}  "
          f"positives={int(y.sum().item())}  negatives={M - int(y.sum().item())}  "
          f"rules with support>0: {int((fired_total > 0).sum())}/{R}")
    return rows, cols, y, fired_total, gold_fired, M


# ---------------------------------------------------------------------------
# Logistic regression fit (per-rule beta + shared intercept)
# ---------------------------------------------------------------------------

def fit_logreg(
    rows: torch.Tensor,
    cols: torch.Tensor,
    y: torch.Tensor,
    M: int,
    R: int,
    device: torch.device,
    l2: float = 1e-4,
    max_iter: int = 200,
    pos_weight: float = None,
    grad_tol: float = 1e-4,
    loss: str = "bce",
):
    """Fit a linear-additive aggregator  score = X @ beta + b  by L-BFGS, with an
    L2 (ridge) penalty and one of three convex, smooth losses. Returns (beta[R], b).

        loss='bce'           binary cross-entropy       -> LOGISTIC REGRESSION
                             (BCEWithLogitsLoss); beta = log-odds contribution,
                             score = logit of P(fact true), calibrated.
        loss='mse'           squared error to {0,1}      -> LINEAR / RIDGE (OLS)
                             (the linear probability model); beta = additive
                             change in a [0,1]-ish score (NOT a probability).
        loss='squared_hinge' squared hinge, labels +/-1  -> LINEAR SVM
                             mean max(0, 1 - s*score)^2, s = 2y-1; beta = margin
                             coefficients, score = signed distance (NOT a prob).

    All three share the SAME additive score form, so in-model --logreg_binary and
    the offline selection/parity stack treat them identically -- this is exactly
    the OLS/SVM-vs-logreg comparison: same design matrix, same L2 grid, same
    ranking machinery, only the loss (hence the MEANING of beta) differs.

    Prints a convergence certificate. Each objective is convex + C^1 (the square
    on the hinge smooths its kink), so the optimum is unique and characterised by
    a ZERO gradient:
      * ||grad||_inf ~ 0  => global penalised optimum reached (grad_tol threshold).
      * iters < max_iter   => L-BFGS stopped on its own tolerance, not the cap.
    Plus coefficient-stability signals (||beta||_2, max|beta|); convex + zero-init
    + deterministic => a re-run gives identical beta.
    """
    if M == 0:
        print("  WARNING: empty design matrix; returning zero weights.")
        return torch.zeros(R), torch.zeros(())
    if loss not in ("bce", "mse", "squared_hinge"):
        raise ValueError(f"fit_logreg: unknown loss {loss!r}")

    indices = torch.stack([rows, cols], dim=0).to(device)
    values  = torch.ones(rows.numel(), dtype=torch.float32, device=device)
    X = torch.sparse_coo_tensor(indices, values, (M, R), device=device).coalesce()
    y = y.to(device)
    s_pm = (2.0 * y - 1.0) if loss == "squared_hinge" else None   # +/-1 labels

    beta = torch.zeros(R, 1, device=device, requires_grad=True)
    b    = torch.zeros(1, device=device, requires_grad=True)

    # pos_weight upweights the positive-class term (ranking-invariant; only
    # rescales beta/calibration). For BCE we use the built-in pos_weight; for
    # mse/hinge we weight the per-row loss the same way.
    have_pw = pos_weight is not None and pos_weight > 0
    pw = torch.tensor(float(pos_weight), device=device) if have_pw else None
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pw) if loss == "bce" else None
    wrow = (torch.where(y > 0.5, torch.full_like(y, float(pos_weight)),
                        torch.ones_like(y)) if (have_pw and loss != "bce") else None)

    def data_loss(logits):
        if loss == "bce":
            return bce(logits, y)
        if loss == "mse":
            e = (logits - y) ** 2
            return (wrow * e).mean() if wrow is not None else e.mean()
        margin = torch.clamp(1.0 - s_pm * logits, min=0.0) ** 2   # squared hinge
        return (wrow * margin).mean() if wrow is not None else margin.mean()

    optimizer = torch.optim.LBFGS(
        [beta, b], lr=1.0, max_iter=max_iter,
        history_size=20, line_search_fn="strong_wolfe")

    state = {"loss": float("nan")}

    def closure():
        optimizer.zero_grad()
        logits = torch.sparse.mm(X, beta).squeeze(1) + b   # [M]
        loss_val = data_loss(logits) + l2 * (beta * beta).sum()
        loss_val.backward()
        state["loss"] = float(loss_val.item())
        return loss_val

    optimizer.step(closure)

    # ---- Convergence certificate --------------------------------------------
    # Iterations actually taken (vs the cap) and the gradient at the solution.
    try:
        n_iter = int(optimizer.state[optimizer._params[0]].get("n_iter", -1))
    except Exception:
        n_iter = -1
    closure()                                   # repopulate .grad at the solution
    grad_inf = max(float(beta.grad.abs().max()), float(b.grad.abs().max()))
    hit_cap  = (n_iter >= max_iter) if n_iter >= 0 else False
    converged = (grad_inf < grad_tol) and not hit_cap

    with torch.no_grad():
        logits = torch.sparse.mm(X, beta).squeeze(1) + b
        # Decision threshold + per-class score summary depend on the loss: BCE
        # thresholds logits at 0 (prob 0.5) and summarises sigmoid probs; MSE
        # thresholds the [0,1] target at 0.5 and summarises the raw prediction;
        # hinge thresholds the decision value at 0 and summarises it.
        if loss == "mse":
            pred = (logits > 0.5).float(); summ = logits; slabel = "mean_pred"
        elif loss == "squared_hinge":
            pred = (logits > 0).float();   summ = logits; slabel = "mean_score"
        else:
            pred = (logits > 0).float();   summ = torch.sigmoid(logits); slabel = "mean_p"
        acc = float((pred == y).float().mean().item())
        sp = float(summ[y > 0.5].mean().item()) if (y > 0.5).any() else float("nan")
        sn = float(summ[y < 0.5].mean().item()) if (y < 0.5).any() else float("nan")
        bnorm = float(beta.norm())
        bmax  = float(beta.abs().max())
        bval  = float(b.item())

    verdict = "CONVERGED" if converged else "NOT CONVERGED"
    print(f"  L-BFGS[{loss}] {verdict}  iters={n_iter}/{max_iter}  "
          f"final_loss={state['loss']:.5f}  ||grad||inf={grad_inf:.2e}")
    print(f"         train_acc={acc:.4f}  {slabel}(pos)={sp:.4f}  {slabel}(neg)={sn:.4f}  "
          f"||beta||2={bnorm:.3f}  max|beta|={bmax:.3f}  intercept={bval:+.3f}")
    if hit_cap:
        print(f"         WARNING: L-BFGS hit the {max_iter}-iter cap without "
              f"meeting its tolerance; raise --max_iter.")
    if grad_inf >= grad_tol and not hit_cap:
        print(f"         WARNING: ||grad||inf={grad_inf:.2e} >= grad_tol={grad_tol:.0e}; "
              f"not at a stationary point (try more iterations / larger --l2).")
    if bmax > 50.0:
        sep = "possible (quasi-)separation; " if loss == "bce" else ""
        print(f"         WARNING: max|beta|={bmax:.1f} is large -> {sep}the "
              f"coefficient is set by L2, not the data. Increase --l2.")
    return beta.detach().squeeze(1).cpu(), b.detach().squeeze().cpu()


# ---------------------------------------------------------------------------
# Argument parsing + main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data_path", default="data/umls",
                   help="Dataset directory (entities.dict, relations.dict, train.txt, ...).")
    p.add_argument("--rule_file", default=None,
                   help="mined_rules.txt (defaults to <data_path>/mined_rules.txt).")
    p.add_argument("--out_dir", default="outputs/additive_umls_aggregation_2x2/logreg",
                   help="Where to write rule_logreg.pt and rule_logreg_train.csv.")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                   help="Device for grounding and the L-BFGS fit. Falls back to "
                        "cpu if CUDA is unavailable.")
    p.add_argument("--chunk_size", type=int, default=0,
                   help="Process each relation's known-subject heads in blocks "
                        "of this size to bound the [B, N] mask/message memory "
                        "(useful for WN18RR/FB15k). 0 = no chunking (default).")
    p.add_argument("--design_cache", default=None,
                   help="Path to a design_matrix.pt cache. The LOO firing matrix "
                        "is DETERMINISTIC given (data, rules), so: if this file "
                        "exists, load it and SKIP grounding (re-sweeping the "
                        "L2 grid then costs only the cheap refits -- matters on "
                        "WN18RR/FB15k where grounding dominates); if it does not "
                        "exist, build the matrix and SAVE it here. Same schema as "
                        "scripts/build_design_matrix_cache.py, so the two share caches.")
    p.add_argument("--loss", default="bce",
                   choices=["bce", "mse", "squared_hinge"],
                   help="Training loss (all share the SAME additive score form, so "
                        "this is the logreg-vs-OLS-vs-SVM comparison; only the "
                        "MEANING of beta changes). 'bce' (default) = LOGISTIC "
                        "REGRESSION (beta = log-odds, calibrated). 'mse' = LINEAR/"
                        "OLS regression to {0,1} (beta = additive score, not a "
                        "probability). 'squared_hinge' = LINEAR SVM (beta = margin "
                        "coefficients).")
    p.add_argument("--l2", type=float, default=1e-4,
                   help="L2 penalty on beta (ridge) for the PRIMARY saved beta "
                        "(the one loaded by --logreg_binary). Keeps never/rarely-"
                        "fired rules near 0 and stabilises the fit. Default 1e-4.")
    p.add_argument("--l2_grid", type=float, nargs="+", default=None,
                   help="Optional log grid of L2 values to sweep OFFLINE, e.g. "
                        "--l2_grid 1e-5 1e-4 1e-3 1e-2 1e-1 1. Fits one beta per "
                        "value (design matrix built once, refit per L2) and saves "
                        "them as beta_grid [G, R] + l2_grid in rule_logreg.pt for "
                        "scripts/select_logreg.py to sweep against alpha and select "
                        "on combined (rule+KGE) validation MRR. The primary --l2 "
                        "beta is always included in the grid.")
    p.add_argument("--max_iter", type=int, default=200,
                   help="L-BFGS max iterations (logistic regression is convex; "
                        "200 is ample). Default 200.")
    p.add_argument("--grad_tol", type=float, default=1e-4,
                   help="Gradient inf-norm threshold for the CONVERGED verdict "
                        "printed by the fit. The convex optimum has zero "
                        "gradient, so ||grad||inf below this certifies the global "
                        "penalised MLE was reached. Default 1e-4.")
    p.add_argument("--pos_weight", type=float, default=0.0,
                   help="Optional BCE positive-class weight (neg/pos imbalance "
                        "correction). 0 = natural MLE (default). Ranking is "
                        "invariant to this; it only rescales beta/calibration.")
    p.add_argument("--top_k", type=int, default=20,
                   help="How many highest-|beta| rules to print for inspection.")
    return p.parse_args()


def main():
    args = parse_args()
    os.chdir(_REPO_ROOT)
    os.makedirs(args.out_dir, exist_ok=True)

    rule_file = args.rule_file or os.path.join(args.data_path, "mined_rules.txt")
    dataset   = os.path.basename(os.path.normpath(args.data_path))

    if args.device == "cuda" and not torch.cuda.is_available():
        print("  WARNING: --device cuda requested but CUDA is unavailable; "
              "falling back to cpu.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"dataset    : {dataset}")
    print(f"data_path  : {args.data_path}")
    print(f"rule_file  : {rule_file}")
    print(f"out_dir    : {args.out_dir}")
    print(f"device     : {device}  chunk_size: {args.chunk_size or 'none'}")
    _model = {"bce": "logistic regression", "mse": "linear/OLS",
              "squared_hinge": "linear SVM"}[args.loss]
    print(f"loss       : {args.loss}  ({_model})")
    print(f"penalty    : l2  l2: {args.l2}  max_iter: {args.max_iter}  "
          f"pos_weight: {args.pos_weight or 'natural'}")

    # ---- Graph + rules -------------------------------------------------------
    print("\nLoading graph...")
    graph = MinimalGraph(args.data_path)
    NR = graph.relation_size
    N  = graph.entity_size
    print(f"  entities={N}  directed_relations={NR * 2}  "
          f"train_facts={len(graph.train_facts)}")

    print("Loading rules...")
    rules = load_rules(rule_file, NR)
    R = len(rules)
    print(f"  num_rules={R}")

    id_to_body   = {rule[0]: rule[2:] for rule in rules}
    lengths_meta = np.array([len(r[2:]) for r in rules], dtype=np.int64)
    heads_meta   = np.array([r[1]       for r in rules], dtype=np.int64)

    # ---- Gold dict -----------------------------------------------------------
    print("\nBuilding gold dict from train facts...")
    gold = build_gold(graph)
    total_known = sum(len(v) for v in gold.values())
    print(f"  unique (r, h) pairs with >=1 gold tail: {total_known}")

    # ---- Design matrix: load from cache or build (grounding) -----------------
    # The LOO firing matrix is deterministic given (data, rules), so a cache is an
    # exact stand-in for re-grounding. This makes an L2 grid EXTENSION cheap:
    # ground once, then every re-sweep is pure-CPU refits.
    cache_path = args.design_cache
    if cache_path and os.path.isfile(cache_path):
        print(f"\nLoading cached design matrix from {cache_path} (skipping grounding)...")
        dm = torch.load(cache_path, map_location="cpu")
        if int(dm["R"]) != R:
            raise SystemExit(
                f"design cache R={int(dm['R'])} != num_rules {R}: the cache was "
                f"built for a different rule file. Delete it or point --design_cache "
                f"elsewhere.")
        rows        = dm["rows"]
        cols        = dm["cols"]
        y           = dm["y"]
        M           = int(dm["M"])
        fired_total = dm["support"].numpy()
        gold_fired  = dm["gold_fired"].numpy()
        print(f"  M={M}  nnz={rows.numel()}  positives={int(y.sum().item())}  "
              f"rules with support>0: {int((fired_total > 0).sum())}/{R}")
    else:
        print("\nBuilding leave-one-out design matrix (grounding from known heads)...")
        rows, cols, y, fired_total, gold_fired, M = build_design_matrix(
            graph, rules, gold, device=device, chunk_size=args.chunk_size)
        if cache_path:
            os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
            torch.save({
                "rows": rows, "cols": cols, "y": y, "M": int(M), "R": int(R),
                "support":    torch.tensor(fired_total, dtype=torch.long),
                "gold_fired": torch.tensor(gold_fired, dtype=torch.long),
                "head":       torch.tensor(heads_meta, dtype=torch.long),
                "length":     torch.tensor(lengths_meta, dtype=torch.long),
                "meta": {"data_path": args.data_path, "rule_file": rule_file,
                         "num_rows": int(M), "num_rules": int(R)},
            }, cache_path)
            print(f"  cached design matrix -> {cache_path} (re-sweeps now skip grounding)")

    # ---- Fit logistic regression (single L2 or a log grid) --------------------
    # The design matrix is built once above; each L2 is a cheap refit. The
    # primary --l2 is always included so `beta` (loaded by --logreg_binary)
    # stays well defined. beta_grid rows line up with l2_grid for the offline
    # alpha/L2 selection in scripts/select_logreg.py.
    pw = (args.pos_weight if args.pos_weight > 0 else None)
    primary = args.l2
    grid    = args.l2_grid
    strengths = sorted(set(list(grid) + [primary])) if grid else [primary]

    model_name = {"bce": "logistic regression", "mse": "linear/OLS regression",
                  "squared_hinge": "linear SVM"}[args.loss]
    print(f"\nFitting per-rule {model_name} (L-BFGS, loss={args.loss}, L2)...")
    beta_rows, intercepts = [], []
    for s in strengths:
        print(f"  L2={s:g}:")
        beta_s, b_s = fit_logreg(
            rows, cols, y, M, R, device=device,
            l2=s, max_iter=args.max_iter, pos_weight=pw, grad_tol=args.grad_tol,
            loss=args.loss)
        beta_rows.append(beta_s.numpy().astype(np.float32))
        intercepts.append(float(b_s.item()))

    beta_grid_np = np.stack(beta_rows, axis=0)          # [G, R]
    l2_grid_np   = np.asarray(strengths, dtype=np.float64)
    primary_idx  = strengths.index(primary)
    beta_np      = beta_grid_np[primary_idx]            # primary beta
    b_val        = intercepts[primary_idx]

    # Regularization path: a coefficient-STABILITY check. ||beta|| should shrink
    # monotonically as the L2 strength grows; a non-monotone jump signals an
    # unstable fit.
    if len(strengths) > 1:
        print(f"\n  Regularization path (||beta|| must shrink monotonically with L2):")
        print(f"    {'L2':>10}  {'||beta||2':>10}  {'max|beta|':>10}  {'nnz':>8}")
        prev = None
        for s, brow in zip(strengths, beta_rows):
            bn = float(np.linalg.norm(brow)); bm = float(np.abs(brow).max())
            nnz = int((brow != 0).sum())
            flag = "  <-- NOT monotone" if (prev is not None and bn > prev * 1.02 + 1e-4) else ""
            print(f"    {s:>10g}  {bn:>10.3f}  {bm:>10.3f}  {nnz:>8}{flag}")
            prev = bn

    n_fired = int((fired_total > 0).sum())
    if n_fired > 0:
        bf = beta_np[fired_total > 0]
        print(f"\n  intercept b = {b_val:+.4f}")
        print(f"  beta (rules with support>0)  "
              f"range=[{bf.min():+.4f}, {bf.max():+.4f}]  "
              f"mean={bf.mean():+.4f}  "
              f"negative={int((bf < 0).sum())}/{n_fired}")

    # ---- Top-|beta| rules for inspection -------------------------------------
    if args.top_k > 0 and n_fired > 0:
        order = np.argsort(-np.abs(beta_np))
        shown = [g for g in order if fired_total[g] > 0][:args.top_k]
        print(f"\n  Top {len(shown)} rules by |beta|:")
        print(f"    {'rule_id':>7}  {'head':>4}  {'len':>3}  {'beta':>9}  "
              f"{'support':>8}  {'prec':>6}  body")
        for gid in shown:
            sup  = int(fired_total[gid])
            prec = (gold_fired[gid] / sup) if sup > 0 else float("nan")
            body = " ".join(str(x) for x in id_to_body.get(gid, []))
            print(f"    {gid:>7}  {int(heads_meta[gid]):>4}  "
                  f"{int(lengths_meta[gid]):>3}  {beta_np[gid]:>+9.4f}  "
                  f"{sup:>8}  {prec:>6.3f}  {body}")

    # ---- Save rule_logreg.pt -------------------------------------------------
    out_pt = os.path.join(args.out_dir, "rule_logreg.pt")
    torch.save({
        "beta":       torch.tensor(beta_np,      dtype=torch.float32),  # primary (at --l2)
        "intercept":  torch.tensor(b_val,        dtype=torch.float32),
        "beta_grid":  torch.tensor(beta_grid_np, dtype=torch.float32),  # [G, R]
        "l2_grid":    torch.tensor(l2_grid_np,   dtype=torch.float64),  # [G]
        "intercept_grid": torch.tensor(np.asarray(intercepts), dtype=torch.float32),  # [G]
        "support":    torch.tensor(fired_total,  dtype=torch.long),
        "gold_fired": torch.tensor(gold_fired,   dtype=torch.long),
        "head":       torch.tensor(heads_meta,   dtype=torch.long),
        "length":     torch.tensor(lengths_meta, dtype=torch.long),
        "meta": {
            "dataset":     dataset,
            "num_rules":   R,
            "num_rows":    M,
            "loss":        args.loss,        # bce=logreg, mse=OLS, squared_hinge=SVM
            "l2":          args.l2,
            "l2_grid":     [float(v) for v in strengths],
            "primary_idx": primary_idx,
            "pos_weight":  args.pos_weight,
            "leave_one_out": True,
        },
    }, out_pt)
    print(f"\nSaved {out_pt}  (beta_grid: {beta_grid_np.shape[0]} L2 values)")

    # ---- Per-rule CSV --------------------------------------------------------
    out_csv = os.path.join(args.out_dir, "rule_logreg_train.csv")
    fieldnames = ["rule_id", "head", "length", "body",
                  "fired_total", "gold_fired", "precision", "beta", "intercept"]
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for gid in range(R):
            sup  = int(fired_total[gid])
            prec = (gold_fired[gid] / sup) if sup > 0 else float("nan")
            body = id_to_body.get(gid, [])
            writer.writerow({
                "rule_id":     gid,
                "head":        int(heads_meta[gid]),
                "length":      int(lengths_meta[gid]),
                "body":        " ".join(str(x) for x in body),
                "fired_total": sup,
                "gold_fired":  int(gold_fired[gid]),
                "precision":   f"{prec:.6f}" if math.isfinite(prec) else "nan",
                "beta":        f"{beta_np[gid]:.6f}",
                "intercept":   f"{b_val:.6f}",
            })
    print(f"Saved {out_csv}  ({R} rows)")
    print("\nDone.")


if __name__ == "__main__":
    main()
