#!/usr/bin/env python3
"""
Unified evaluator for original and GAT backbones.

This script ensures apples-to-apples evaluation by using identical ranking and
metric logic across variants, while only changing score combination strategy.
It can also train an external adaptive-beta combiner on the validation split.
"""

import argparse
import hashlib
import importlib
import json
import os
import random
import sys
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified evaluator for RulE variants")
    parser.add_argument("--project_root", type=str, required=True)
    parser.add_argument("--backend", type=str, required=True, choices=["original", "gat"])
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=800)
    parser.add_argument("--train_adaptive", action="store_true", default=False)
    parser.add_argument("--adaptive_epochs", type=int, default=5)
    parser.add_argument("--adaptive_lr", type=float, default=0.01)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def import_backend_modules(project_root: str, backend: str):
    src_dir = (
        os.path.join(project_root, "RulE_original", "src")
        if backend == "original"
        else os.path.join(project_root, "src")
    )
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    data_mod = importlib.import_module("data")
    model_mod = importlib.import_module("model")
    utils_mod = importlib.import_module("utils")
    return data_mod, model_mod, utils_mod


def resolve_path(config_path: str, maybe_relative_path: str) -> str:
    if os.path.isabs(maybe_relative_path):
        return maybe_relative_path
    return os.path.normpath(os.path.join(os.path.dirname(config_path), maybe_relative_path))


def load_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    elif "model" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    alpha: float,
    beta_rel: torch.Tensor = None,
    beta_density: torch.Tensor = None,
) -> Dict[str, float]:
    model.eval()
    ranks = []

    for batch in dataloader:
        all_h, all_r, all_t, flag = batch
        all_h = all_h.squeeze(0).to(device)
        all_r = all_r.squeeze(0).to(device)
        all_t = all_t.squeeze(0).to(device)
        flag = flag.squeeze(0).to(device)

        rule_score, mask = model(all_h, all_r, None)
        kge_score = model.compute_g_KGE(all_h, all_r)

        if beta_rel is None:
            logits = rule_score + alpha * kge_score
        else:
            min_score = rule_score.min(dim=-1, keepdim=True)[0]
            grounded = ((rule_score - min_score) > 1e-6).float()
            density = grounded.mean(dim=-1)
            beta = torch.sigmoid(beta_rel[all_r] + beta_density * density).unsqueeze(-1)
            logits = beta * rule_score + (1.0 - beta) * kge_score

        for i in range(all_t.size(0)):
            t = all_t[i].item()
            if mask[i, t].item():
                val = logits[i, t]
                L = (logits[i][flag[i]] > val).sum().item() + 1
                H = (logits[i][flag[i]] >= val).sum().item() + 2
            else:
                L = 1
                H = model.num_entities + 1
            ranks.append((L, H))

    hit1 = hit3 = hit10 = mr = mrr = 0.0
    for L, H in ranks:
        for rank in range(L, H):
            w = 1.0 / (H - L)
            if rank <= 1:
                hit1 += w
            if rank <= 3:
                hit3 += w
            if rank <= 10:
                hit10 += w
            mr += rank * w
            mrr += (1.0 / rank) * w

    n = max(1, len(ranks))
    return {
        "Hit@1": hit1 / n,
        "Hit@3": hit3 / n,
        "Hit@10": hit10 / n,
        "MR": mr / n,
        "MRR": mrr / n,
    }


def train_adaptive_beta(
    model: nn.Module,
    valid_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    beta_rel = nn.Parameter(torch.zeros(model.num_relations * 2, device=device))
    beta_density = nn.Parameter(torch.zeros(1, device=device))
    optimizer = torch.optim.Adam([beta_rel, beta_density], lr=lr)

    best_mrr = -1.0
    best_state = None

    for _ in range(epochs):
        model.eval()
        for batch in valid_loader:
            all_h, all_r, all_t, flag = batch
            all_h = all_h.squeeze(0).to(device)
            all_r = all_r.squeeze(0).to(device)
            all_t = all_t.squeeze(0).to(device)
            flag = flag.squeeze(0).to(device)

            with torch.no_grad():
                rule_score, mask = model(all_h, all_r, None)
                kge_score = model.compute_g_KGE(all_h, all_r)

            min_score = rule_score.min(dim=-1, keepdim=True)[0]
            grounded = ((rule_score - min_score) > 1e-6).float()
            density = grounded.mean(dim=-1)
            beta = torch.sigmoid(beta_rel[all_r] + beta_density * density).unsqueeze(-1)
            logits = beta * rule_score + (1.0 - beta) * kge_score

            # True-vs-negative margin objective on filtered negatives.
            losses = []
            for i in range(all_h.size(0)):
                if not mask[i, all_t[i]].item():
                    continue
                pos = logits[i, all_t[i]]
                neg_mask = flag[i].clone()
                neg_mask[all_t[i]] = False
                neg_idx = torch.where(neg_mask)[0]
                if neg_idx.numel() == 0:
                    continue
                if neg_idx.numel() > 128:
                    perm = torch.randperm(neg_idx.numel(), device=device)[:128]
                    neg_idx = neg_idx[perm]
                neg_scores = logits[i, neg_idx]
                losses.append(F.relu(1.0 - pos + neg_scores).mean())

            if not losses:
                continue

            loss = torch.stack(losses).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        metrics = evaluate(
            model=model,
            dataloader=valid_loader,
            device=device,
            alpha=0.0,
            beta_rel=beta_rel.detach(),
            beta_density=beta_density.detach(),
        )
        if metrics["MRR"] > best_mrr:
            best_mrr = metrics["MRR"]
            best_state = (beta_rel.detach().clone(), beta_density.detach().clone(), metrics)

    if best_state is None:
        return beta_rel.detach(), beta_density.detach(), {"MRR": 0.0}
    return best_state


def config_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_mod, model_mod, utils_mod = import_backend_modules(args.project_root, args.backend)
    configs = utils_mod.load_config(args.config)
    cfg = configs[0] if isinstance(configs, (list, tuple)) else configs

    graph = data_mod.KnowledgeGraph(resolve_path(args.config, cfg.data_path))

    model = model_mod.RulE(
        graph=graph,
        p_norm=cfg.p_norm,
        mlp_rule_dim=cfg.mlp_rule_dim,
        gamma_fact=cfg.gamma_fact,
        gamma_rule=cfg.gamma_rule,
        hidden_dim=cfg.hidden_dim,
        device=device,
        dataset=cfg.data_path,
    )

    rule_file = resolve_path(args.config, cfg.rule_file)
    rule_negative_size = getattr(cfg, "rule_negative_size", 64)
    ruleset = data_mod.RuleDataset(graph.relation_size, rule_file, rule_negative_size)
    rules = [rule[0] for rule in ruleset.rules]
    model.set_rules(rules)

    load_checkpoint(model, args.checkpoint, device)
    model = model.to(device)
    model.eval_compute_rule_weight(device)

    batch_size = getattr(cfg, "g_batch_size", 16)
    valid_loader = DataLoader(data_mod.ValidDataset(graph, batch_size), batch_size=1, num_workers=0)
    test_loader = DataLoader(data_mod.TestDataset(graph, batch_size), batch_size=1, num_workers=0)

    fixed_metrics = evaluate(model, test_loader, device, alpha=args.alpha)

    adaptive_metrics = None
    beta_rel = beta_density = None
    valid_adaptive = None
    if args.train_adaptive:
        beta_rel, beta_density, valid_adaptive = train_adaptive_beta(
            model=model,
            valid_loader=valid_loader,
            device=device,
            epochs=args.adaptive_epochs,
            lr=args.adaptive_lr,
        )
        adaptive_metrics = evaluate(
            model=model,
            dataloader=test_loader,
            device=device,
            alpha=0.0,
            beta_rel=beta_rel,
            beta_density=beta_density,
        )

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    payload = {
        "backend": args.backend,
        "config": os.path.abspath(args.config),
        "config_hash": config_hash(args.config),
        "checkpoint": os.path.abspath(args.checkpoint),
        "seed": args.seed,
        "alpha": args.alpha,
        "fixed_metrics": fixed_metrics,
        "adaptive_metrics": adaptive_metrics,
        "adaptive_valid_metrics": valid_adaptive,
        "beta_density": None if beta_density is None else float(beta_density.item()),
        "beta_mean": None if beta_rel is None else float(torch.sigmoid(beta_rel).mean().item()),
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
