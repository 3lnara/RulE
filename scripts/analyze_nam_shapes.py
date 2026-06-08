#!/usr/bin/env python3
"""Visualise the per-rule NAM shape functions learned by a frozen-linear + NAM run.

For each rule R the model has learned a shape function

    f_R(count) = nam_net( [count, log1p(count), z_R] )

and a frozen linear weight w_R (RulE confidence). The combined per-rule
contribution to a candidate tail t is

    contrib_R(t) = w_R * count_R[t] + f_R(count_R[t])

This script evaluates both curves over a count grid [0 .. max_count] and either
prints a text table or saves matplotlib figures.

Usage
-----
    # Text table of top-K rules by firing frequency:
    python analyze_nam_shapes.py --save_path /path/to/outputs/additive_umls_nam_fm \
        --rule_file /path/to/data/umls/mined_rules.txt \
        --top_k 20

    # Save PDF/PNG plots for a specific set of rule IDs:
    python analyze_nam_shapes.py --save_path /path/to/outputs/additive_umls_nam_fm \
        --rule_file /path/to/data/umls/mined_rules.txt \
        --rule_ids 0,5,42 --plot --out_dir ./plots
"""

import argparse
import os
import sys

import numpy as np


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def read_rules(rule_file):
    """Return list of [head, body...] int lists (rule i = i-th line)."""
    rules = []
    with open(rule_file) as f:
        for line in f:
            toks = line.strip().split()
            if len(toks) > 1:
                rules.append([int(t) for t in toks])
    return rules


def load_checkpoint(save_path):
    """Load grounding.pt and return state_dict."""
    import torch
    ckpt = os.path.join(save_path, 'grounding.pt')
    if not os.path.exists(ckpt):
        raise FileNotFoundError("Checkpoint not found: %s" % ckpt)
    state = torch.load(ckpt, map_location='cpu')
    return state.get('model', state)


def load_nam_params(state_dict):
    """Extract nam_emb [num_rules, nam_dim] and nam_net weights from state_dict.

    Returns (nam_emb_np, layers) where layers is a list of (W, b) numpy pairs
    representing the MLP in forward order.
    """
    import torch

    if 'nam_emb' not in state_dict:
        raise KeyError(
            "'nam_emb' not in checkpoint -- was this run trained with --nam?")

    nam_emb = state_dict['nam_emb'].detach().cpu().numpy()   # [num_rules, nam_dim]

    # Collect all MLP layers: keys like 'nam_net.layers.0.weight'. The layer
    # index is the token right before the trailing '.weight' (use [-2] so this
    # is robust to the module-prefix depth).
    layer_keys = sorted(
        [k for k in state_dict if k.startswith('nam_net.layers.') and k.endswith('.weight')],
        key=lambda k: int(k.split('.')[-2])
    )
    layers = []
    for wk in layer_keys:
        bk = wk.replace('.weight', '.bias')
        W = state_dict[wk].detach().cpu().numpy()   # [out, in]
        b = state_dict[bk].detach().cpu().numpy()   # [out]
        layers.append((W, b))

    return nam_emb, layers


def load_rule_weights(state_dict):
    """Return sigmoid(rule_weight_logit) as a numpy array [num_rules]."""
    import torch
    logit = state_dict['rule_weight_logit']
    return torch.sigmoid(logit).detach().cpu().numpy()


def relu(x):
    return np.maximum(x, 0.0)


def mlp_forward(x, layers):
    """Forward pass through a (W, b) layer list with ReLU on all but last."""
    out = x
    for i, (W, b) in enumerate(layers):
        out = out @ W.T + b
        if i < len(layers) - 1:
            out = relu(out)
    return out


def eval_shape(rule_id, count_grid, nam_emb, layers):
    """Evaluate f_R(count) = nam_net([count, log1p(count), z_R]) over count_grid.

    count_grid: 1-D array of count values (e.g. np.arange(0, max_count+1))
    Returns array of same length.
    """
    z = nam_emb[rule_id]                        # [nam_dim]
    n = len(count_grid)
    phi = np.stack(
        [count_grid.astype(np.float32),
         np.log1p(count_grid.astype(np.float32))],
        axis=-1,
    )                                            # [n, 2]
    z_tile = np.tile(z[None, :], (n, 1))        # [n, nam_dim]
    inp = np.concatenate([phi, z_tile], axis=-1) # [n, 2+nam_dim]
    return mlp_forward(inp, layers).squeeze(-1)  # [n]


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #

def mode_print(args, rules, state_dict):
    """Print a text table of shape function statistics for top-K rules."""
    nam_emb, layers = load_nam_params(state_dict)
    w = load_rule_weights(state_dict)

    num_rules = nam_emb.shape[0]
    max_count = int(args.max_count)
    count_grid = np.arange(0, max_count + 1, dtype=np.float32)

    # Determine which rule IDs to show.
    if args.rule_ids:
        rule_ids = [int(r) for r in args.rule_ids.split(',')]
    else:
        # Use the top-K rules by |w_R| (proxy for importance).
        top_k = min(int(args.top_k), num_rules)
        rule_ids = list(np.argsort(-np.abs(w))[:top_k])

    relation_names = None
    if args.relation_names:
        try:
            relation_names = open(args.relation_names).read().splitlines()
        except Exception:
            pass

    def rel_name(rid):
        if relation_names and rid < len(relation_names):
            return relation_names[rid]
        return str(rid)

    header = ('%-6s  %-30s  %8s  %10s  %10s  %10s  %10s'
              % ('RuleID', 'Rule (head <- body...)', 'w_R',
                 'NAM@0', 'NAM@1', 'NAM@5', 'NAM@10'))
    print(header)
    print('-' * len(header))

    for rid in rule_ids:
        if rid >= len(rules) or rid >= num_rules:
            continue
        rule = rules[rid]
        head_id = rule[0]
        body_ids = rule[1:]
        rule_str = ('%s <- %s' % (rel_name(head_id),
                                   ', '.join(rel_name(b) for b in body_ids)))
        nam_vals = eval_shape(rid, count_grid, nam_emb, layers)
        print('%-6d  %-30s  %8.4f  %10.4f  %10.4f  %10.4f  %10.4f'
              % (rid, rule_str[:30], w[rid],
                 nam_vals[0] if 0  <= max_count else 0,
                 nam_vals[1] if 1  <= max_count else 0,
                 nam_vals[5] if 5  <= max_count else 0,
                 nam_vals[10] if 10 <= max_count else 0))


def mode_plot(args, rules, state_dict):
    """Save per-rule contribution curve plots to --out_dir."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    nam_emb, layers = load_nam_params(state_dict)
    w = load_rule_weights(state_dict)

    num_rules = nam_emb.shape[0]
    max_count = int(args.max_count)
    count_grid = np.arange(0, max_count + 1, dtype=np.float32)

    if args.rule_ids:
        rule_ids = [int(r) for r in args.rule_ids.split(',')]
    else:
        top_k = min(int(args.top_k), num_rules)
        rule_ids = list(np.argsort(-np.abs(w))[:top_k])

    relation_names = None
    if args.relation_names:
        try:
            relation_names = open(args.relation_names).read().splitlines()
        except Exception:
            pass

    def rel_name(rid):
        if relation_names and rid < len(relation_names):
            return relation_names[rid]
        return 'rel_%d' % rid

    out_dir = args.out_dir or '.'
    os.makedirs(out_dir, exist_ok=True)

    for rid in rule_ids:
        if rid >= len(rules) or rid >= num_rules:
            continue
        rule = rules[rid]
        head_id = rule[0]
        body_ids = rule[1:]

        linear_curve = w[rid] * count_grid
        nam_curve    = eval_shape(rid, count_grid, nam_emb, layers)
        combined     = linear_curve + nam_curve

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

        ax1.set_title('NAM shape f_R(count)  [rule %d]' % rid)
        ax1.plot(count_grid, nam_curve, color='steelblue', label='f_R (NAM residual)')
        ax1.axhline(0, color='black', linewidth=0.5)
        ax1.set_xlabel('count_R')
        ax1.set_ylabel('f_R(count)')
        ax1.legend(fontsize=8)

        ax2.set_title('Combined contribution  [rule %d]' % rid)
        ax2.plot(count_grid, linear_curve, color='grey',     linestyle='--',
                 label='w_R * count  (linear)')
        ax2.plot(count_grid, nam_curve,    color='steelblue', linestyle=':',
                 label='f_R(count)  (NAM residual)')
        ax2.plot(count_grid, combined,     color='firebrick',
                 label='combined = w_R*count + f_R')
        ax2.axhline(0, color='black', linewidth=0.5)
        ax2.set_xlabel('count_R')
        ax2.set_ylabel('contribution to score[t]')
        ax2.legend(fontsize=8)

        rule_desc = '%s <- %s' % (
            rel_name(head_id), ', '.join(rel_name(b) for b in body_ids))
        fig.suptitle('Rule %d: %s' % (rid, rule_desc[:60]), fontsize=9)
        fig.tight_layout()

        out_file = os.path.join(out_dir, 'rule_%05d.pdf' % rid)
        fig.savefig(out_file, bbox_inches='tight')
        plt.close(fig)
        print('Saved %s' % out_file)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description='Visualise per-rule NAM shape functions from a frozen-linear + NAM run.')
    parser.add_argument('--save_path', required=True,
                        help='Directory containing grounding.pt (model checkpoint).')
    parser.add_argument('--rule_file', required=True,
                        help='Path to mined_rules.txt for the dataset.')
    parser.add_argument('--rule_ids', type=str, default=None,
                        help='Comma-separated rule IDs to analyse. '
                             'If omitted, top --top_k rules by |w_R| are used.')
    parser.add_argument('--top_k', type=int, default=20,
                        help='Number of top rules to show when --rule_ids is not set.')
    parser.add_argument('--max_count', type=int, default=20,
                        help='Evaluate shape functions over count grid [0 .. max_count].')
    parser.add_argument('--relation_names', type=str, default=None,
                        help='Optional text file with one relation name per line '
                             '(line i = relation id i). Used for readable rule labels.')
    parser.add_argument('--plot', action='store_true', default=False,
                        help='Save matplotlib PDF plots to --out_dir. '
                             'Without this flag the table is printed to stdout.')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Output directory for plots (default: current directory).')
    args = parser.parse_args()

    rules = read_rules(args.rule_file)
    state_dict = load_checkpoint(args.save_path)

    if args.plot:
        mode_plot(args, rules, state_dict)
    else:
        mode_print(args, rules, state_dict)


if __name__ == '__main__':
    main()
