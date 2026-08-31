#!/usr/bin/env python3
"""Dump per-(query, rule, entity) raw integer path counts for valid + test.

One grounding pass over the frozen pretrain embeddings. Writes
counts_{valid,test}.pt and rule_meta.pt; --dump_kge adds
kge_{valid,test}.pt.

Usage:
    python scripts/dump_rule_counts.py --checkpoint <grounding.pt> \\
        --config <config.json> --data_path data/<ds> \\
        --rule_file data/<ds>/mined_rules.txt --out_dir outputs/<ds>/rq --dump_kge
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import torch
import torch.nn as nn

# src_additive on sys.path for model.py + layers.py only (not data.py: torch_scatter).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.normpath(os.path.join(_SCRIPT_DIR, ".."))
_SRC_DIR    = os.path.join(_REPO_ROOT, "src_additive")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import model as _model_mod          # noqa: E402  (model.py, no torch_scatter)
RulE = _model_mod.RulE


# Minimal graph: CSR adjacency from the triple files, no torch_scatter.
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
        """Single-hop message passing over `relation`: [N, B] counts -> [N, B]."""
        src, dst = self.relation2adjacency[relation]
        if src.numel() == 0:
            return torch.zeros_like(x)
        # Keep the adjacency on the same device as the state tensor x.
        if src.device != x.device:
            src = src.to(x.device)
            dst = dst.to(x.device)
            self.relation2adjacency[relation] = (src, dst)
        N, B = x.size()
        messages = x[src]                            # [E, B]
        new_x    = torch.zeros(N, B, dtype=x.dtype, device=x.device)
        new_x.scatter_add_(0, dst.unsqueeze(1).expand_as(messages), messages)
        return new_x

    def grounding(self, h: torch.Tensor, r_head: int,
                  r_body: list, edges_to_remove=None) -> torch.Tensor:
        """Ground rule body `r_body` from heads h [B] -> [B, N] path counts
        (edges_to_remove ignored in the dump context)."""
        N = self.entity_size
        B = h.size(0)
        x = torch.zeros(N, B, dtype=torch.long, device=h.device)
        x.scatter_(0, h.unsqueeze(0), 1)          # one-hot init [N, B]
        for rb in r_body:
            x = self.propagate(x, rb)
        return x.transpose(0, 1)                   # [B, N]


# ---------------------------------------------------------------------------
# Rule loading (reads mined_rules.txt directly; no RuleDataset needed)
# ---------------------------------------------------------------------------

def load_rules(rule_file: str, num_relations: int):
    """List of [global_id, r_head, b1, b2, ...]; global_id is a sequential counter
    over kept lines (>=2 tokens), matching src_additive/data.py RuleDataset."""
    rules = []
    gid = 0
    with open(rule_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            rules.append([gid] + [int(p) for p in parts])
            gid += 1
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
               device: torch.device, chunk: int = 0) -> dict:
    """Ground every query in split_name -> CSR count structure. `chunk` caps how
    many queries of a relation are grounded at once (0 = all; use ~512 for WN18RR/FB15k)."""
    facts = getattr(graph, f"{split_name}_facts")   # list of (h, r, t)
    Q = len(facts)
    query_h    = torch.tensor([h for h, _, _ in facts], dtype=torch.long)
    query_r    = torch.tensor([r for _, r, _ in facts], dtype=torch.long)
    query_gold = torch.tensor([t for _, _, t in facts], dtype=torch.long)

    # Per-relation query groups (preserve global query index qi).
    rel2queries = defaultdict(list)   # r -> [(qi, h)]
    for qi, (h, r, t) in enumerate(facts):
        rel2queries[r].append((qi, h))

    qi_parts, rule_parts, ent_parts, cnt_parts = [], [], [], []

    t0 = time.time()
    for r, qlist in sorted(rel2queries.items()):
        rules_for_r = model.relation2rules[r]
        if not rules_for_r:
            continue
        Bn = len(qlist)
        step = Bn if (chunk is None or chunk <= 0) else chunk
        for blk in range(0, Bn, step):
            block = qlist[blk:blk + step]
            h_t  = torch.tensor([h for _, h in block], dtype=torch.long, device=device)
            qi_t = torch.tensor([qi for qi, _ in block], dtype=torch.long, device=device)

            for rule_global_id, (r_head, r_body) in rules_for_r:
                assert r_head == r
                with torch.no_grad():
                    counts = graph.grounding(h_t, r_head, r_body, None).long()  # [b, N]
                    nz = counts.nonzero(as_tuple=False)          # [K, 2] (local_i, ent)
                    if nz.size(0) == 0:
                        continue
                    li   = nz[:, 0]
                    ent  = nz[:, 1]
                    cval = counts[li, ent].clamp(max=32767)
                qi_parts.append(qi_t[li].cpu())
                ent_parts.append(ent.cpu())
                cnt_parts.append(cval.cpu().to(torch.short))
                rule_parts.append(torch.full((nz.size(0),), rule_global_id,
                                             dtype=torch.long))

    elapsed = time.time() - t0
    print(f"  grounding loop done in {elapsed:.1f}s")

    # Concatenate flat firings and group into CSR by query index. Ranking scores
    # scatter_add per query, so within-query order is irrelevant; a plain sort by
    # qi is enough to build q_ptr.
    if qi_parts:
        qi_all   = torch.cat(qi_parts)
        ent_all  = torch.cat(ent_parts)
        cnt_all  = torch.cat(cnt_parts)
        rule_all = torch.cat(rule_parts)
        order    = torch.argsort(qi_all)
        qi_all, ent_all = qi_all[order], ent_all[order]
        cnt_all, rule_all = cnt_all[order], rule_all[order]
        q_ptr = torch.zeros(Q + 1, dtype=torch.long)
        q_ptr[1:] = torch.cumsum(torch.bincount(qi_all, minlength=Q), dim=0)
    else:
        ent_all  = torch.zeros(0, dtype=torch.long)
        cnt_all  = torch.zeros(0, dtype=torch.short)
        rule_all = torch.zeros(0, dtype=torch.long)
        q_ptr    = torch.zeros(Q + 1, dtype=torch.long)

    return {
        "query_h":    query_h,
        "query_r":    query_r,
        "query_gold": query_gold,
        "q_ptr":      q_ptr,
        "rule_id":    rule_all,
        "ent":        ent_all,
        "cnt":        cnt_all,
    }


# ---------------------------------------------------------------------------
# Per-query KGE score dump  (kge[q, t] = model.compute_g_KGE for query q)
# ---------------------------------------------------------------------------

def dump_kge(model, graph: MinimalGraph, split_name: str,
             device: torch.device, chunk: int = 256) -> dict:
    """Dump the frozen RotatE score of every entity for every query ->
    {'query_h','query_r','query_gold','kge'} with kge [Q, N] float16. Row order
    matches counts_<split>.pt."""
    facts = getattr(graph, f"{split_name}_facts")
    Q = len(facts)
    N = graph.entity_size
    query_h = torch.tensor([h for h, _, _ in facts], dtype=torch.long)
    query_r = torch.tensor([r for _, r, _ in facts], dtype=torch.long)
    query_g = torch.tensor([t for _, _, t in facts], dtype=torch.long)

    kge = torch.empty(Q, N, dtype=torch.float16)
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, Q, chunk):
            e = min(s + chunk, Q)
            h = query_h[s:e].to(device)
            r = query_r[s:e].to(device)
            scores = model.compute_g_KGE(h, r)            # [b, N]
            kge[s:e] = scores.detach().to("cpu", torch.float16)
    print(f"  KGE dump done in {time.time() - t0:.1f}s  shape={tuple(kge.shape)}")

    return {
        "query_h":    query_h,
        "query_r":    query_r,
        "query_gold": query_g,
        "kge":        kge,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Dump per-(query, rule, entity) counts for UMLS.")
    p.add_argument("--checkpoint",
                   default="outputs/additive_umls_conf_binary_clamp/grounding.pt")
    p.add_argument("--config",
                   default="outputs/additive_umls_conf_binary_clamp/config.json")
    p.add_argument("--data_path",  default="data/umls")
    p.add_argument("--rule_file",  default="data/umls/mined_rules.txt")
    p.add_argument("--out_dir",
                   default="outputs/additive_umls_aggregation_2x2/analysis")
    p.add_argument("--splits", nargs="+", default=["valid", "test"])
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                   help="Device for grounding AND the KGE score dump. Use 'cuda' "
                        "for large datasets (WN18RR/FB15k); falls back to cpu if "
                        "CUDA is unavailable.")
    p.add_argument("--grounding_chunk", type=int, default=0,
                   help="Queries per grounding block within a relation, bounding "
                        "the [B, N] count tensor. 0 = all queries of the relation "
                        "at once (fine for small N like UMLS); set e.g. 512 for "
                        "WN18RR/FB15k.")
    p.add_argument("--dump_kge", action="store_true", default=False,
                   help="Also dump per-query frozen KGE (RotatE) scores to "
                        "kge_<split>.pt [Q, N] float16, for the offline rule+KGE "
                        "alpha sweep in scripts/select_logreg.py.")
    p.add_argument("--kge_chunk", type=int, default=256,
                   help="Queries per KGE batch. The RotatE tail tensor is "
                        "[kge_chunk, N, dim], so for large N (e.g. WN18RR "
                        "N=40943) use a SMALL value (8-32) to bound memory; the "
                        "default 256 is only safe for small N like UMLS.")
    return p.parse_args()


def main():
    args = parse_args()
    os.chdir(_REPO_ROOT)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("  WARNING: --device cuda requested but CUDA is unavailable; "
              "using cpu.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"device: {device}  grounding_chunk: {args.grounding_chunk or 'none'}")

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
    model.to(device)
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
        "w_R_unclamped": w_unclamped.cpu(),
        "bodies":        bodies,
    }
    meta_path = os.path.join(args.out_dir, "rule_meta.pt")
    torch.save(rule_meta, meta_path)
    print(f"  saved rule_meta.pt  ({len(rules)} rules)")

    # Dump each split (grounding + KGE share `device`; both saved as CPU tensors).
    for split in args.splits:
        print(f"\nDumping split '{split}'...")
        t_start = time.time()
        csr = dump_split(model, graph, split, device, chunk=args.grounding_chunk)
        elapsed = time.time() - t_start
        nnz = csr["rule_id"].numel()
        Q   = csr["query_h"].numel()
        print(f"  queries={Q}  nnz={nnz}  ({elapsed:.1f}s total)")
        out_path = os.path.join(args.out_dir, f"counts_{split}.pt")
        torch.save(csr, out_path)
        print(f"  saved counts_{split}.pt")

        if args.dump_kge:
            kge = dump_kge(model, graph, split, device, chunk=args.kge_chunk)
            kge_path = os.path.join(args.out_dir, f"kge_{split}.pt")
            torch.save(kge, kge_path)
            print(f"  saved kge_{split}.pt")

    print("\nDone.")


if __name__ == "__main__":
    main()
