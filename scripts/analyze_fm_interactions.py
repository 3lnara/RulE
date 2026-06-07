#!/usr/bin/env python3
"""Interpret the learned pairwise FM interactions of a frozen-linear + FM run.

The FM term adds, for each candidate tail t,

    sum_{R < R'} <v_R, v_R'> * 1[c_R[t] > 0] * 1[c_R'[t] > 0]

so the scalar synergy of a rule pair (R, R') is simply the dot product
<v_R, v_R'> of their learned FM embeddings. Two rules can only interact when
they co-fire, and rules only co-fire when they share the same head relation
(they ground candidate tails for the same query). Hence the meaningful synergy
structure is block-diagonal per head relation, and we never need the full
num_rules x num_rules matrix.

Two modes:

  pairs (default, fully offline -- only needs rule_fm_emb + the rule file):
      For each head relation, rank its rule pairs by |<v_R, v_R'>| and dump the
      strongest synergies / anti-synergies.

  decompose (needs the dataset + checkpoint):
      For one query (h, r) and a target tail t, run grounding, then report the
      exact per-pair FM contributions <v_R, v_R'> * b_R[t] * b_R'[t]. These sum
      to the FM score at t, giving an exact "which rule pairs drove this
      prediction" breakdown alongside the linear per-rule contributions.
"""

import argparse
import os
import sys

import numpy as np


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_fm_emb(save_path):
    """Return the FM embedding matrix [num_rules, k] as a numpy array.

    Prefers the convenience dump rule_fm_emb.npy; falls back to extracting the
    'rule_fm_emb' tensor from grounding.pt.
    """
    npy_path = os.path.join(save_path, 'rule_fm_emb.npy')
    if os.path.exists(npy_path):
        return np.load(npy_path)

    ckpt_path = os.path.join(save_path, 'grounding.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            "Found neither %s nor %s." % (npy_path, ckpt_path))

    import torch
    state = torch.load(ckpt_path, map_location='cpu')
    model_state = state.get('model', state)
    if 'rule_fm_emb' not in model_state:
        raise KeyError(
            "'rule_fm_emb' not in %s -- was this run trained with "
            "--fm_interactions?" % ckpt_path)
    return model_state['rule_fm_emb'].detach().cpu().numpy()


def read_rules(rule_file):
    """Reproduce RuleDataset's ordering: rule index i is the i-th non-trivial
    line of the rule file. Returns a list of [head, body...] (relation ids),
    indexed by rule id.
    """
    rules = []
    with open(rule_file, 'r') as fi:
        for line in fi:
            toks = line.strip().split()
            if len(toks) <= 1:
                continue
            ints = [int(t) for t in toks]
            # Line layout matches RuleDataset: first token is the head
            # relation, the remainder is the body.
            rules.append(ints)
    return rules


def load_relation_names(data_path):
    """id2relation dict from relations.dict, or None if unavailable."""
    if not data_path:
        return None
    rel_dict = os.path.join(data_path, 'relations.dict')
    if not os.path.exists(rel_dict):
        return None
    id2rel = {}
    with open(rel_dict) as fi:
        for line in fi:
            rid, rname = line.strip().split('\t')
            id2rel[int(rid)] = rname
    return id2rel


def relation_name(rid, id2rel, num_base_relations):
    """Human-readable relation label, marking inverse relations with _inv."""
    if id2rel is None or num_base_relations is None:
        return str(rid)
    if rid < num_base_relations:
        return id2rel.get(rid, str(rid))
    base = rid - num_base_relations
    return '%s_inv' % id2rel.get(base, str(base))


def body_str(body, id2rel, num_base_relations):
    return ' -> '.join(relation_name(b, id2rel, num_base_relations) for b in body)


# --------------------------------------------------------------------------- #
# Mode: per-relation top synergy pairs
# --------------------------------------------------------------------------- #
def build_relation2rules(rules):
    """head relation id -> list of rule ids whose head is that relation."""
    rel2rules = {}
    for idx, r in enumerate(rules):
        head = r[0]
        rel2rules.setdefault(head, []).append(idx)
    return rel2rules


def analyze_pairs(args):
    V = load_fm_emb(args.save_path)              # [num_rules, k]
    rules = read_rules(args.rule_file)
    if len(rules) != V.shape[0]:
        print('[warn] rule_file has %d rules but FM embedding has %d rows; '
              'indices may be misaligned.' % (len(rules), V.shape[0]),
              file=sys.stderr)

    id2rel = load_relation_names(args.data_path)
    num_base = len(id2rel) if id2rel is not None else None

    rel2rules = build_relation2rules(rules)

    out_path = args.out or os.path.join(args.save_path, 'fm_top_pairs.tsv')
    fout = open(out_path, 'w')
    fout.write('head_relation\trule_i\trule_j\tsynergy\t'
               'rule_i_body\trule_j_body\n')

    # Sort relations by how many rules they own (most structure first).
    rel_order = sorted(rel2rules.items(), key=lambda kv: -len(kv[1]))

    for head, rule_ids in rel_order:
        if len(rule_ids) < 2:
            continue
        ids = np.asarray(rule_ids)
        Vr = V[ids]                              # [m, k]
        G = Vr @ Vr.T                            # [m, m] synergy matrix
        m = G.shape[0]
        iu = np.triu_indices(m, k=1)             # upper triangle, no diagonal
        synergies = G[iu]
        order = np.argsort(-np.abs(synergies))   # by |synergy| descending
        top = order[:args.top_k]

        head_label = relation_name(head, id2rel, num_base)
        print('\n=== head relation %s (%d rules) ===' % (head_label, m))
        for o in top:
            ri = int(ids[iu[0][o]])
            rj = int(ids[iu[1][o]])
            s = float(synergies[o])
            bi = body_str(rules[ri][1:], id2rel, num_base)
            bj = body_str(rules[rj][1:], id2rel, num_base)
            print('  %+.4f  rule %d [%s]  x  rule %d [%s]' % (s, ri, bi, rj, bj))
            fout.write('%s\t%d\t%d\t%.6f\t%s\t%s\n'
                       % (head_label, ri, rj, s, bi, bj))

    fout.close()
    print('\nWrote top pairs to %s' % out_path)


# --------------------------------------------------------------------------- #
# Mode: per-prediction decomposition
# --------------------------------------------------------------------------- #
def decompose_query(args):
    """Load the full model + dataset, run grounding for (h, r), and report the
    exact linear and FM (pairwise) contributions to the score at target t.
    """
    import torch

    # Make the src_additive package importable.
    src_dir = args.src_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src_additive')
    sys.path.insert(0, src_dir)

    from data import KnowledgeGraph, RuleDataset          # noqa: E402
    from model import RulE                                 # noqa: E402
    from utils import load_config                          # noqa: E402

    cfg = load_config(args.config)[0]
    device = torch.device('cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu')

    graph = KnowledgeGraph(cfg.data_path)
    ruleset = RuleDataset(graph.relation_size, cfg.rule_file, cfg.rule_negative_size)
    rule_list = [rule[0] for rule in ruleset.rules]

    model = RulE(graph, cfg.p_norm, cfg.mlp_rule_dim, cfg.gamma_fact,
                 cfg.gamma_rule, cfg.hidden_dim, device, cfg.data_path)
    model.set_rules(rule_list)

    # Recreate the additive/FM parameters so the checkpoint loads cleanly.
    ckpt = torch.load(os.path.join(args.save_path, 'grounding.pt'), map_location=device)
    model_state = ckpt.get('model', ckpt)
    num_rules = model.rule_features.size(0)
    if 'rule_weight_logit' in model_state:
        model.register_buffer('rule_weight_logit',
                              torch.zeros(num_rules, device=device))
        model.simple_aggregation = True
    if 'rule_fm_emb' in model_state:
        k = model_state['rule_fm_emb'].shape[1]
        import torch.nn as nn
        model.rule_fm_emb = nn.Parameter(torch.zeros(num_rules, k, device=device))
        model.use_fm = True
    model.load_state_dict(model_state, strict=False)
    model = model.to(device)
    model.eval()

    h, r, t = args.decompose
    all_h = torch.tensor([h], dtype=torch.long, device=device)

    # Reproduce the grounding loop for this single query.
    rule_ids, counts = [], []
    with torch.no_grad():
        for index, (r_head, r_body) in model.relation2rules[r]:
            c = graph.grounding(all_h, r_head, r_body, None).float()   # [1, entities]
            rule_ids.append(index)
            counts.append(c.squeeze(0))
    if not rule_ids:
        print('No rules fire for relation %d.' % r)
        return

    rule_ids = torch.tensor(rule_ids, dtype=torch.long, device=device)
    C = torch.stack(counts, dim=0)                      # [R, entities]
    b = (C[:, t] > 0).float()                           # [R] co-fire at target t

    # Linear per-rule contribution at t.
    conf = torch.sigmoid(model.rule_weight_logit[rule_ids])
    lin = conf * C[:, t]                                # [R]

    print('\nQuery (h=%d, r=%d, target t=%d)' % (h, r, t))
    print('Rules firing: %d | co-firing at t: %d'
          % (rule_ids.numel(), int(b.sum().item())))

    # FM pairwise contributions at t (only over co-firing rules).
    if getattr(model, 'use_fm', False):
        active = torch.nonzero(b > 0, as_tuple=True)[0]
        V = model.rule_fm_emb[rule_ids][active]         # [a, k]
        ids_active = rule_ids[active]
        G = V @ V.t()                                   # [a, a] = pair contributions (b=1)
        a = G.shape[0]
        pairs = []
        for i in range(a):
            for j in range(i + 1, a):
                pairs.append((float(G[i, j]),
                              int(ids_active[i]), int(ids_active[j])))
        pairs.sort(key=lambda x: -abs(x[0]))
        fm_total = sum(p[0] for p in pairs)
        print('\nFM score at t = %.4f  (sum of %d pair contributions)'
              % (fm_total, len(pairs)))
        print('Top pair contributions:')
        for s, ri, rj in pairs[:args.top_k]:
            print('  %+.4f  rule %d x rule %d' % (s, ri, rj))

    lin_total = float(lin.sum().item())
    print('\nLinear score at t = %.4f' % lin_total)
    topk = torch.argsort(-lin)[:args.top_k]
    print('Top linear per-rule contributions:')
    for o in topk.tolist():
        print('  %+.4f  rule %d  (count=%.0f, conf=%.3f)'
              % (float(lin[o]), int(rule_ids[o]),
                 float(C[o, t]), float(conf[o])))


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--save_path', required=True,
                    help='Run directory containing rule_fm_emb.npy / grounding.pt.')
    ap.add_argument('--rule_file', help='mined_rules.txt (required for pairs mode).')
    ap.add_argument('--data_path', default=None,
                    help='Dataset dir (for relations.dict -> readable names).')
    ap.add_argument('--top_k', type=int, default=20,
                    help='How many pairs to report per relation / per query.')
    ap.add_argument('--out', default=None,
                    help='Output TSV path (pairs mode). Default: <save_path>/fm_top_pairs.tsv')

    ap.add_argument('--decompose', type=int, nargs=3, metavar=('H', 'R', 'T'),
                    default=None,
                    help='Decompose the prediction for query (H, R) at target T.')
    ap.add_argument('--config', default=None,
                    help='Config json (required for --decompose).')
    ap.add_argument('--src_dir', default=None,
                    help='Path to src_additive (decompose mode); auto-detected by default.')
    ap.add_argument('--cuda', action='store_true', default=False)
    args = ap.parse_args()

    if args.decompose is not None:
        if not args.config:
            ap.error('--decompose requires --config.')
        decompose_query(args)
    else:
        if not args.rule_file:
            ap.error('pairs mode requires --rule_file.')
        analyze_pairs(args)


if __name__ == '__main__':
    main()
