"""
Compare fixed alpha vs learnable beta for KGE/Rule score balancing.

Usage:
    python compare_balancing.py --config ../config/family_config.json --checkpoint ../outputs/your_model/grounding.pt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from tqdm import tqdm
from torch.utils.data import DataLoader

from data import KnowledgeGraph, TestDataset, ValidDataset
from model import RulE
from utils import load_config


def parse_args():
    parser = argparse.ArgumentParser(description='Compare balancing methods')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='test', choices=['valid', 'test'])
    parser.add_argument('--alpha', type=float, default=5.0, help='Fixed alpha value')
    parser.add_argument('--train_beta', action='store_true', help='Fine-tune learnable beta')
    parser.add_argument('--beta_epochs', type=int, default=5, help='Epochs to train beta')
    parser.add_argument('--beta_lr', type=float, default=0.01, help='Learning rate for beta')
    return parser.parse_args()


def evaluate_method(model, dataloader, device, method='fixed_alpha', alpha=5.0):
    """Evaluate using a specific balancing method."""
    model.eval()
    
    all_ranks = []
    weight_infos = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating ({method})"):
            all_h, all_r, all_t, flag = batch
            all_h = all_h.squeeze(0).to(device)
            all_r = all_r.squeeze(0).to(device)
            all_t = all_t.squeeze(0).to(device)
            flag = flag.squeeze(0).to(device)
            
            # Get rule grounding score
            rule_score, mask = model(all_h, all_r, None)
            
            # Get KGE score
            kge_score = model.compute_g_KGE(all_h, all_r)
            
            # Combine using specified method
            combined_score, weight_info = model.combine_scores(
                rule_score, kge_score, all_r, 
                method=method, alpha=alpha
            )
            weight_infos.append(weight_info)
            
            # Compute ranks
            for k in range(all_t.size(0)):
                t = all_t[k].item()
                if mask[k, t].item():
                    val = combined_score[k, t]
                    L = (combined_score[k][flag[k]] > val).sum().item() + 1
                    H = (combined_score[k][flag[k]] >= val).sum().item() + 2
                else:
                    L = 1
                    H = model.num_entities + 1
                all_ranks.append((L, H))
    
    # Compute metrics (expectation)
    hit1, hit3, hit10, mr, mrr = 0.0, 0.0, 0.0, 0.0, 0.0
    for L, H in all_ranks:
        for rank in range(L, H):
            weight = 1.0 / (H - L)
            if rank <= 1:
                hit1 += weight
            if rank <= 3:
                hit3 += weight
            if rank <= 10:
                hit10 += weight
            mr += rank * weight
            mrr += (1.0 / rank) * weight
    
    n = len(all_ranks)
    metrics = {
        'hit@1': hit1 / n,
        'hit@3': hit3 / n,
        'hit@10': hit10 / n,
        'mr': mr / n,
        'mrr': mrr / n,
    }
    
    return metrics, weight_infos


def train_beta(model, train_dataloader, valid_dataloader, device, epochs=5, lr=0.01):
    """Fine-tune the learnable beta parameters."""
    print("\n" + "="*60)
    print("Training learnable beta parameters...")
    print("="*60)
    
    # Only train beta parameters
    for param in model.parameters():
        param.requires_grad = False
    model.beta.requires_grad = True
    
    optimizer = torch.optim.Adam([model.beta], lr=lr)
    
    best_mrr = 0
    best_beta = model.beta.data.clone()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            all_h, all_r, all_t, flag = batch
            all_h = all_h.squeeze(0).to(device)
            all_r = all_r.squeeze(0).to(device)
            all_t = all_t.squeeze(0).to(device)
            flag = flag.squeeze(0).to(device)
            
            optimizer.zero_grad()
            
            # Get scores
            rule_score, mask = model(all_h, all_r, None)
            kge_score = model.compute_g_KGE(all_h, all_r)
            
            # Combine with learnable beta
            combined_score, _ = model.combine_scores(
                rule_score, kge_score, all_r, 
                method='learnable_beta_unnorm'
            )
            
            # Margin ranking loss: correct answer should score higher
            batch_size = all_t.size(0)
            losses = []
            for k in range(batch_size):
                t = all_t[k]
                pos_score = combined_score[k, t]
                
                # Sample negative entities
                neg_mask = flag[k].clone()
                neg_mask[t] = False
                neg_indices = torch.where(neg_mask)[0]
                
                if len(neg_indices) > 0:
                    # Sample up to 10 negatives
                    n_neg = min(10, len(neg_indices))
                    neg_sample = neg_indices[torch.randperm(len(neg_indices))[:n_neg]]
                    neg_scores = combined_score[k, neg_sample]
                    
                    # Margin ranking loss
                    margin = 1.0
                    loss = F.relu(margin - pos_score + neg_scores).mean()
                    losses.append(loss)
            
            if losses:
                loss = torch.stack(losses).mean()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # Validate
        metrics, _ = evaluate_method(model, valid_dataloader, device, 
                                     method='learnable_beta_unnorm')
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Valid MRR={metrics['mrr']:.4f}")
        
        if metrics['mrr'] > best_mrr:
            best_mrr = metrics['mrr']
            best_beta = model.beta.data.clone()
    
    # Restore best beta
    model.beta.data = best_beta
    
    # Re-enable all gradients
    for param in model.parameters():
        param.requires_grad = True
    
    return best_mrr


def print_results(method_name, metrics):
    """Pretty print evaluation results."""
    print(f"\n{'='*60}")
    print(f"Results for: {method_name}")
    print('='*60)
    print(f"  Hit@1:  {metrics['hit@1']:.4f}")
    print(f"  Hit@3:  {metrics['hit@3']:.4f}")
    print(f"  Hit@10: {metrics['hit@10']:.4f}")
    print(f"  MR:     {metrics['mr']:.2f}")
    print(f"  MRR:    {metrics['mrr']:.4f}")


def main():
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load config
    config = load_config(args.config)
    if isinstance(config, (list, tuple)):
        config = config[0]
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() and config.cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    data_path = config.data_path
    # Handle relative paths
    if not os.path.isabs(data_path):
        data_path = os.path.join(os.path.dirname(args.config), data_path)
    # Normalize path
    data_path = os.path.normpath(data_path)
    print(f"Data path: {data_path}")
    graph = KnowledgeGraph(data_path)
    
    # Load test/valid set
    # TestDataset needs batch_size, and uses test_facts from graph
    # We need to temporarily swap which facts are used based on split
    batch_size = getattr(config, 'g_batch_size', 8)
    
    test_set = TestDataset(graph, batch_size)
    test_dataloader = DataLoader(test_set, batch_size=1, num_workers=0)
    
    # For validation, use ValidDataset
    valid_set = ValidDataset(graph, batch_size)
    valid_dataloader = DataLoader(valid_set, batch_size=1, num_workers=0)
    
    # Load model
    print("Loading model...")
    model = RulE(
        graph=graph,
        p_norm=config.p_norm,
        mlp_rule_dim=config.mlp_rule_dim,
        gamma_fact=config.gamma_fact,
        gamma_rule=config.gamma_rule,
        hidden_dim=config.hidden_dim,
        device=device,
        dataset=config.data_path
    )
    
    # Load rules
    rules = []
    rule_file = config.rule_file
    # Handle relative paths
    if not os.path.isabs(rule_file):
        rule_file = os.path.join(os.path.dirname(args.config), rule_file)
    
    print(f"Loading rules from: {rule_file}")
    with open(rule_file, 'r') as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            if parts:  # Skip empty lines
                rule = [i] + [int(x) for x in parts]
                rules.append(rule)
    print(f"Loaded {len(rules)} rules")
    model.set_rules(rules)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model = model.to(device)
    
    # Compute rule weights
    model.eval_compute_rule_weight(device)
    
    print("\n" + "="*60)
    print("COMPARISON: Fixed Alpha vs Learnable Beta")
    print("="*60)
    
    # === Method 1: Fixed Alpha (Original) ===
    print(f"\n[1] Evaluating with fixed alpha = {args.alpha}...")
    metrics_fixed, _ = evaluate_method(
        model, test_dataloader, device, 
        method='fixed_alpha', alpha=args.alpha
    )
    print_results(f"Fixed Alpha (α={args.alpha})", metrics_fixed)
    
    # === Method 2: Learnable Beta (untrained) ===
    print(f"\n[2] Evaluating with learnable beta (untrained, β=0.5)...")
    metrics_beta_untrained, weight_info = evaluate_method(
        model, test_dataloader, device,
        method='learnable_beta_unnorm'
    )
    print_results("Learnable Beta (untrained)", metrics_beta_untrained)
    
    # === Method 3: Learnable Beta (trained) ===
    if args.train_beta:
        print(f"\n[3] Training learnable beta for {args.beta_epochs} epochs...")
        
        # Use valid set for training beta, test set for final eval
        train_beta(model, valid_dataloader, valid_dataloader, device,
                  epochs=args.beta_epochs, lr=args.beta_lr)
        
        # Print learned beta values
        beta_summary = model.get_beta_summary()
        print(f"\nLearned beta summary:")
        print(f"  Mean rule weight: {beta_summary['mean_rule_weight']:.4f}")
        print(f"  Mean KGE weight:  {beta_summary['mean_kge_weight']:.4f}")
        print(f"  Beta range: [{beta_summary['min_beta']:.4f}, {beta_summary['max_beta']:.4f}]")
        
        # Final evaluation
        print(f"\n[3] Evaluating with learnable beta (trained)...")
        metrics_beta_trained, _ = evaluate_method(
            model, test_dataloader, device,
            method='learnable_beta_unnorm'
        )
        print_results("Learnable Beta (trained)", metrics_beta_trained)
    
    # === Summary ===
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)
    print(f"{'Method':<30} {'MRR':>10} {'Hit@1':>10} {'Hit@10':>10}")
    print("-"*60)
    print(f"{'Fixed Alpha (α=' + str(args.alpha) + ')':<30} {metrics_fixed['mrr']:>10.4f} {metrics_fixed['hit@1']:>10.4f} {metrics_fixed['hit@10']:>10.4f}")
    print(f"{'Learnable Beta (untrained)':<30} {metrics_beta_untrained['mrr']:>10.4f} {metrics_beta_untrained['hit@1']:>10.4f} {metrics_beta_untrained['hit@10']:>10.4f}")
    if args.train_beta:
        print(f"{'Learnable Beta (trained)':<30} {metrics_beta_trained['mrr']:>10.4f} {metrics_beta_trained['hit@1']:>10.4f} {metrics_beta_trained['hit@10']:>10.4f}")
    
    # Relative improvement
    print("\n" + "-"*60)
    if args.train_beta:
        improvement = (metrics_beta_trained['mrr'] - metrics_fixed['mrr']) / metrics_fixed['mrr'] * 100
        print(f"MRR change (trained beta vs fixed alpha): {improvement:+.2f}%")


if __name__ == '__main__':
    main()
