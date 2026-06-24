#!/usr/bin/env python3
"""
Dump per-(query, rule, entity) raw integer counts for UMLS (valid + test splits).

One grounding pass over the frozen pretrain embeddings; no variant-specific weights
needed. Outputs are fully variant-agnostic: binary-vs-raw and keep-vs-clamp scoring
all derive from these counts + rule_meta.pt offline.

Does NOT import data.py (which requires torch_scatter), so this script runs
locally with a standard PyTorch installation. Grounding is reimplemented using
torch.scatter_add_ (pure PyTorch).

Outputs (torch.save) under --out_dir
(default outputs/additive_umls_aggregation_2x2/analysis/):

  counts_valid.pt  / counts_test.pt
    query_h  : LongTensor [Q]
    query_r  : LongTensor [Q]
    query_gold: LongTensor [Q]
    q_ptr    : LongTensor [Q+1]   CSR row-pointer into rule_id/ent/cnt
    rule_id  : LongTensor [nnz]   global rule index (0..num_rules-1)
    ent      : LongTensor [nnz]   entity index
    cnt      : ShortTensor [nnz]  raw path count (int16, capped at 32767)

  rule_meta.pt
    head          : LongTensor [R]
    length        : LongTensor [R]      body length
    w_R_unclamped : FloatTensor [R]     gamma_rule - ||body_emb - head_emb||
    bodies        : list of lists[int]  (variable length, not padded)

Usage (run from repo root):
    python scripts/dump_rule_counts.py \\
        --checkpoint outputs/additive_umls_paper_sum_clamp/grounding.pt \\
        --config     outputs/additive_umls_paper_sum_clamp/config.json \\
        --data_path  data/umls \\
        --rule_file  data/umls/mined_rules.txt \\
        --out_dir    outputs/additive_umls_aggregation_2x2/analysis
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Bootstrap: add src_additive to sys.path for model.py + layers.py only.
# We do NOT import data.py because it requires torch_scatter.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.normpath(os.path.join(_SCRIPT_DIR, ".."))
_SRC_DIR    = os.path.join(_REPO_ROOT, "src_additive")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import model as _model_mod          # noqa: E402  (model.py, no torch_scatter)
RulE = _model_mod.RulE


# ---------------------------------------------------------------------------
# Minimal graph: reads triples from files, builds CSR adjacency without
# torch_scatter (uses pure Python lists + torch.scatter_add_).
# ---------------------------------------------------------------------------

class MinimalGraph:
    """Reads KG files; provides .grounding() without torch_scatter."""

    def __init__(self, data_path: str):
        self.data_path = data_path

        self.entity2id  = {}
        self.relation2id = {}
        with open(os.path.join(data_path, "entities.dict")) as f:
            for line in f:
                idx, name = line.strip().split("\t")
                self.entity2id[name] = int(idx)
        with open(os.path.join(data_path, "relations.dict")) as f:
            for line in f:
                idx, name = line.strip().split("\t")
                self.relation2id[name] = int(idx)

        self.entity_size   = len(self.entity2id)
        self.relation_size = len(self.relation2id)
        N  = self.entity_size
        NR = self.relation_size

        # Adjacency: for each directed relation (forward r and inverse r+NR),
        # store src_list and dst_list (will be converted to LongTensors).
        src_lists = [[] for _ in range(NR * 2)]
        dst_lists = [[] for _ in range(NR * 2)]

        self.train_facts = []
        self.valid_facts = []
        self.test_facts  = []

        def _add_edge(r, h, t):
            src_lists[r].append(h)
            dst_lists[r].append(t)
            src_lists[r + NR].append(t)
            dst_lists[r + NR].append(h)

        def _read_split(fname, store):
            with open(fname) as f:
                for line in f:
                    h, r, t = line.strip().split("\t")
                    h = self.entity2id[h]
                    r = self.relation2id[r]
                    t = self.entity2id[t]
                    store.append((h, r, t))
                    store.append((t, r + NR, h))

        def _read_train(fname, store):
            with open(fname) as f:
                for line in f:
                    h, r, t = line.strip().split("\t")
                    h = self.entity2id[h]
                    r = self.relation2id[r]
                    t = self.entity2id[t]
                    store.append((h, r, t))
                    store.append((t, r + NR, h))
                    _add_edge(r, h, t)

        _read_train(os.path.join(data_path, "train.txt"), self.train_facts)
        _read_split(os.path.join(data_path, "valid.txt"), self.valid_facts)
        _read_split(os.path.join(data_path, "test.txt"),  self.test_facts)

        # Convert adjacency to LongTensors.
        # relation2adjacency[r] = (src_tensor, dst_tensor)
        self.relation2adjacency = []
        for r in range(NR * 2):
            src = torch.tensor(src_lists[r], dtype=torch.long)
            dst = torch.tensor(dst_lists[r], dtype=torch.long)
            self.relation2adjacency.append((src, dst))

    def propagate(self, x: torch.Tensor, relation: int) -> torch.Tensor:
        """Single-hop message passing (no edge removal).

        x: [N, B] integer counts (source state)
        Returns updated [N, B].

        Replaces data.KnowledgeGraph.propagate (which uses torch_scatter)
        with torch.scatter_add_ (standard PyTorch).
        """
        src, dst = self.relation2adjacency[relation]
        if src.numel() == 0:
            return torch.zeros_like(x)
        N, B = x.size()
        messages = x[src]                            # [E, B]
        new_x    = torch.zeros(N, B, dtype=x.dtype)
        new_x.scatter_add_(0, dst.unsqueeze(1).expand_as(messages), messages)
        return new_x

    def grounding(self, h: torch.Tensor, r_head: int,
                  r_body: list, edges_to_remove=None) -> torch.Tensor:
        """Ground a single rule body from all heads in h.

        h: [B] head entity indices
        Returns [B, N] integer count tensor (paths from each h to each entity).
        edges_to_remove: ignored (always None in eval / dump context).
        """
        N = self.entity_size
        B = h.size(0)
        x = torch.zeros(N, B, dtype=torch.long)
        x.scatter_(0, h.unsqueeze(0), 1)          # one-hot init [N, B]
        for rb in r_body:
            x = self.propagate(x, rb)
        return x.transpose(0, 1)                   # [B, N]


# ---------------------------------------------------------------------------
# Rule loading (reads mined_rules.txt directly; no RuleDataset needed)
# ---------------------------------------------------------------------------

def load_rules(rule_file: str, num_relations: int):
    """Returns list of [global_id, r_head, b1, b2, ...] (no padding)."""
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

def build_model(cfg: dict, graph: MinimalGraph, rules: list, device: torch.device):
    """Build RulE shell, register rule_weight_logit buffer, load checkpoint."""

    class _FakeGraph:
        """Minimal graph wrapper for RulE.__init__ (entity/relation sizes only)."""
        entity_size   = graph.entity_size
        relation_size = graph.relation_size

    model = RulE(
        graph=_FakeGraph(),
        p_norm=int(cfg.get("p_norm", 2)),
        mlp_rule_dim=int(cfg.get("mlp_rule_dim", 100)),
        gamma_fact=float(cfg.get("gamma_fact", 6)),
        gamma_rule=float(cfg.get("gamma_rule", 8)),
        hidden_dim=int(cfg.get("hidden_dim", 2000)),
        device=device,
        dataset=cfg.get("data_path", "umls"),
    )
    model.set_rules(rules)
    dummy = torch.zeros(model.rule_features.size(0), device=device)
    model.register_buffer("rule_weight_logit", dummy)
    return model


# ---------------------------------------------------------------------------
# Unclamped weight computation (mirrors trainer.py lines 405-418, no clamp)
# ---------------------------------------------------------------------------

def compute_unclamped_weights(model, device: torch.device) -> torch.Tensor:
    rule_features = model.rule_features.to(device)
    rule_masks    = model.rule_masks.to(device)
    batch     = 128
    split_num = max(1, rule_features.size(0) // batch)
    logits = []
    with torch.no_grad():
        for rules_b, masks_b in zip(
            torch.split(rule_features, split_num, 0),
            torch.split(rule_masks,    split_num, 0),
        ):
            scores, _ = model.add_ruleE(rules_b.unsqueeze(1), masks_b)
            logits.append(scores.squeeze(1))
    return torch.cat(logits, dim=0).detach().float()


# ---------------------------------------------------------------------------
# Grounding dump for one split
# ---------------------------------------------------------------------------

def dump_split(model, graph: MinimalGraph, split_name: str,
               device: torch.device) -> dict:
    """Ground every query in split_name; return CSR count structure."""
    facts = getattr(graph, f"{split_name}_facts")   # list of (h, r, t)

    # Build per-relation query groups to batch the grounding call.
    # Use the SAME ordering as facts so we can build a flat query list.
    rel2queries = defaultdict(list)   # r -> [(query_idx, h, t)]
    query_h    = []
    query_r    = []
    query_gold = []
    for qi, (h, r, t) in enumerate(facts):
        query_h.append(h)
        query_r.append(r)
        query_gold.append(t)
        rel2queries[r].append((qi, h, t))

    Q = len(facts)
    per_query_entries = [[] for _ in range(Q)]   # (rule_id, ent, cnt)

    t0 = time.time()
    for r, qlist in sorted(rel2queries.items()):
        rules_for_r = model.relation2rules[r]
        if not rules_for_r:
            continue

        all_h_list = [h for (_, h, _) in qlist]
        all_h_t    = torch.tensor(all_h_list, dtype=torch.long, device=device)
        local2qi   = [qi for (qi, _, _) in qlist]

        for rule_global_id, (r_head, r_body) in rules_for_r:
            assert r_head == r
            with torch.no_grad():
                counts = graph.grounding(all_h_t, r_head, r_body, None)
                counts = counts.long()    # [batch, N]

            for local_i, qi in enumerate(local2qi):
                row = counts[local_i]                          # [N]
                nz  = row.nonzero(as_tuple=True)[0]
                if nz.numel() == 0:
                    continue
                for e_idx, c_val in zip(nz.tolist(), row[nz].tolist()):
                    per_query_entries[qi].append(
                        (rule_global_id, e_idx, min(c_val, 32767))
                    )

    elapsed = time.time() - t0
    print(f"  grounding loop done in {elapsed:.1f}s")

    # Pack into CSR
    q_ptr_list   = [0]
    rule_id_list = []
    ent_list     = []
    cnt_list     = []
    for qi in range(Q):
        for (rid, e, c) in per_query_entries[qi]:
            rule_id_list.append(rid)
            ent_list.append(e)
            cnt_list.append(c)
        q_ptr_list.append(len(rule_id_list))

    return {
        "query_h":    torch.tensor(query_h,    dtype=torch.long),
        "query_r":    torch.tensor(query_r,    dtype=torch.long),
        "query_gold": torch.tensor(query_gold, dtype=torch.long),
        "q_ptr":      torch.tensor(q_ptr_list, dtype=torch.long),
        "rule_id":    torch.tensor(rule_id_list, dtype=torch.long),
        "ent":        torch.tensor(ent_list,   dtype=torch.long),
        "cnt":        torch.tensor(cnt_list,   dtype=torch.short),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Dump per-(query, rule, entity) counts for UMLS.")
    p.add_argument("--checkpoint",
                   default="outputs/additive_umls_paper_sum_clamp/grounding.pt")
    p.add_argument("--config",
                   default="outputs/additive_umls_paper_sum_clamp/config.json")
    p.add_argument("--data_path",  default="data/umls")
    p.add_argument("--rule_file",  default="data/umls/mined_rules.txt")
    p.add_argument("--out_dir",
                   default="outputs/additive_umls_aggregation_2x2/analysis")
    p.add_argument("--splits", nargs="+", default=["valid", "test"])
    return p.parse_args()


def main():
    args = parse_args()
    os.chdir(_REPO_ROOT)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cpu")

    # Config
    with open(args.config) as f:
        cfg = json.load(f)

    # Graph
    print("Loading graph...")
    graph = MinimalGraph(args.data_path)
    print(f"  entities={graph.entity_size}  relations={graph.relation_size}")

    # Rules
    print("Loading rules...")
    rules = load_rules(args.rule_file, graph.relation_size)
    print(f"  num_rules={len(rules)}")

    # Model
    print("Building model + loading checkpoint...")
    model = build_model(cfg, graph, rules, device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"  loaded from {args.checkpoint}")

    # Unclamped w_R
    print("Computing unclamped rule weights...")
    w_unclamped = compute_unclamped_weights(model, device)
    n_neg = int((w_unclamped < 0).sum().item())
    print(f"  range [{w_unclamped.min():.3f}, {w_unclamped.max():.3f}]  "
          f"negative={n_neg}/{len(rules)}")

    # Rule metadata (bodies from relation2rules — unpadded, safe)
    print("Building rule_meta...")
    id_to_body = {}
    for r_idx in range(graph.relation_size * 2):
        for global_id, (r_head, r_body) in model.relation2rules[r_idx]:
            id_to_body[global_id] = (r_head, list(r_body))

    heads, lengths, bodies = [], [], []
    for i in range(model.rule_features.size(0)):
        gid = model.rule_features[i, 0].item()
        r_head, body = id_to_body[gid]
        heads.append(r_head)
        lengths.append(len(body))
        bodies.append(body)

    rule_meta = {
        "head":          torch.tensor(heads,   dtype=torch.long),
        "length":        torch.tensor(lengths, dtype=torch.long),
        "w_R_unclamped": w_unclamped,
        "bodies":        bodies,
    }
    meta_path = os.path.join(args.out_dir, "rule_meta.pt")
    torch.save(rule_meta, meta_path)
    print(f"  saved rule_meta.pt  ({len(rules)} rules)")

    # Dump each split
    for split in args.splits:
        print(f"\nDumping split '{split}'...")
        t_start = time.time()
        csr = dump_split(model, graph, split, device)
        elapsed = time.time() - t_start
        nnz = csr["rule_id"].numel()
        Q   = csr["query_h"].numel()
        print(f"  queries={Q}  nnz={nnz}  ({elapsed:.1f}s total)")
        out_path = os.path.join(args.out_dir, f"counts_{split}.pt")
        torch.save(csr, out_path)
        print(f"  saved counts_{split}.pt")

    print("\nDone.")


if __name__ == "__main__":
    main()
