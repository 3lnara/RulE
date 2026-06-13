
import logging, os, datetime
import argparse
import torch
from data import KnowledgeGraph, TrainDataset, ValidDataset, TestDataset, RuleDataset, KGETrainDataset
from model import RulE
from utils import load_config, save_config, set_logger, set_seed
from trainer import GroundTrainer, PreTrainer

# torch.cuda.set_device(1)

def save_files(rules):
    with open('mined_rules.txt','w') as fw:
        for rule in rules:
            for relation in rule[0:-1]:
                fw.writelines(str(relation) + ' ')

            fw.writelines(str(rule[-1])+'\n')

def formatted_rules(_rules):
    rules = []
    
    for i, _rule in enumerate(_rules):
        rule = [i,len(_rule)]
        rule += _rule
        rules.append(rule)
    return rules

def parse_args(args=None):

    parser = argparse.ArgumentParser(
        description='RNNLogic',
        usage='train.py [<args>] [-h | --help]'
    )
    parser.add_argument("--local_rank", type=int, default=0)
    # data path
    parser.add_argument('--data_path', default="../data/wn18rr", type=str, help='dataset path')
    parser.add_argument('--rule_file', default="../data/wn18rr/mined_rules.txt", type=str)
    # device 
    parser.add_argument('--cuda', action='store_true',default=False, help='use GPU')
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)

    parser.add_argument('--seed',default=800, type=int, help='seed')
    
    # pre train process (KGE + rulE)
    parser.add_argument('-b', '--batch_size', default=256, type=int)
    parser.add_argument('-n', '--negative_sample_size', default=256 , type=int)
    parser.add_argument('--rule_batch_size',default=128,type=int, help='rule batch size')
    parser.add_argument('--rule_negative_size',default=64,type=int)

    parser.add_argument('-d', '--hidden_dim', default=500, type=int)
    parser.add_argument('-g_f', '--gamma_fact', default=6, type=float, help='the triplet margin')
    parser.add_argument('-g_r', '--gamma_rule', default=5, type=float, help='the rule margin')
    parser.add_argument('--disable_adv', action='store_true',default=True, help='disable the adversarial negative sampling')
    # parser.add_argument('-adv', '--negative_adversarial_sampling', default=True, action='store_true')
    parser.add_argument('-a', '--adversarial_temperature', default=0.5, type=float)
                            
    parser.add_argument('--uni_weight', action='store_true', 
                        help='Otherwise use subsampling weighting like in word2vec')
    parser.add_argument('-lr', '--learning_rate', default=0.00005, type=float)
    parser.add_argument('--warm_up_steps', default=None, type=int)
    parser.add_argument('--g_warm_up_steps', default=None, type=int)
    parser.add_argument('--save_checkpoint_steps', default=10, type=int)
    parser.add_argument('--valid_steps', default=1000, type=int)
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--weight_rule',type=float,default=1)
    parser.add_argument('-reg', '--regularization', default=0, type=float)
    parser.add_argument('--max_steps', default=15000, type=int)
    parser.add_argument('--p_norm', default=2, type=int)

    # save path
    parser.add_argument('-init', '--init_checkpoint_config', default="../config/umls_config.json", type=str)
    parser.add_argument('-save', '--save_path', default=None, type=str)

    
    # grounding training process
  
    parser.add_argument('--mlp_rule_dim', default=100, type=int)
    parser.add_argument('--alpha', default=5.0, type=int, help='weight the KGE score')
    parser.add_argument('--smoothing', default=0.5, type=float)
    parser.add_argument('--batch_per_epoch', default=1000000, type=int)
    parser.add_argument('--print_every', default=1000, type=int)
    parser.add_argument('--g_batch_size', default=16, type=int)
    parser.add_argument('--g_lr', default=0.00005, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--num_iters', default=20, type=int)

    parser.add_argument('--skip_pretrain', action='store_true', default=False,
                        help='Skip pre-training and load from an existing checkpoint')
    parser.add_argument('--pretrain_checkpoint', type=str, default=None,
                        help='Path to pre-trained checkpoint (used with --skip_pretrain)')

    # === Diagnostic flags for grounding stability ===
    parser.add_argument('--use_rule_confidence_variant_a', action='store_true', default=False,
                    help='Use frozen RulE confidences (mean of rules_weight_emb) instead of '
                         'learnable conf_proj during grounding.')

    parser.add_argument('--eval_every_batches', default=0, type=int,
                        help='If >0, run validation every N batches during iteration 1 only. '
                             'Use to track MRR within the first epoch (set e.g. 200 for family). '
                             '0 disables (default).')
    parser.add_argument('--freeze_conf_proj', action='store_true', default=False,
                        help='Freeze grounding_gat.conf_proj (weight + bias) during grounding. '
                             'Diagnostic: if MRR stops degrading, the conf_proj.bias drift is the '
                             'culprit; otherwise the GAT is.')
    parser.add_argument('--attn_entropy_weight', default=0.0, type=float,
                        help='Weight \u03bb for attention entropy regularisation. Adds '
                             '\u03bb * \u03a3 a*log(a) to the loss; with \u03bb>0 this rewards uniform '
                             'attention (\u03a3 a*log(a) is most negative when uniform, ~0 when peaked). '
                             '0 disables (default). Try 1e-4 \u2013 1e-3.')
    parser.add_argument('--attn_dim', default=128, type=int,
                        help='Attention hidden dimension for GroundingGAT. '
                             'Default 128. Use 64 on small graphs (UMLS) to reduce overfitting.')
    parser.add_argument('--checkpoint_grounding', action='store_true', default=False,
                        help='Gradient-checkpoint the GAT edge-attention computation so its '
                             '[num_edges, attn_dim] activations are recomputed in backward '
                             'instead of stored. Cuts grounding memory dramatically (lets large '
                             'graphs like WN18RR fit at attn_dim=128 on a 16 GB GPU) at the cost '
                             'of ~one extra forward per step. Gradients are exact. Disables the '
                             'attention-entropy regulariser (incompatible by construction).')
    parser.add_argument('--gat_variant', default='baseline', type=str,
                        choices=['baseline', 'no_dst', 'rule_attn'],
                        help='GAT architecture for rule grounding. '
                             'baseline: GATv2 with W_src + W_dst + W_rel and a shared attn_vec. '
                             'no_dst: drop W_dst (provably redundant under per-target softmax) '
                             '       -- variant 2-clean. '
                             'rule_attn: drop W_dst and replace the shared attn_vec with a '
                             '          rule-conditioned attn_from_rule(rule_emb) -- Option A.')
    return parser.parse_args(args)

def main():
    args = parse_args()

    skip_pretrain = args.skip_pretrain
    pretrain_checkpoint = args.pretrain_checkpoint
    cli_save_path = args.save_path
    cli_eval_every_batches = args.eval_every_batches
    cli_freeze_conf_proj = args.freeze_conf_proj
    cli_attn_entropy_weight = args.attn_entropy_weight
    cli_use_rule_confidence_variant_a = args.use_rule_confidence_variant_a
    cli_attn_dim = args.attn_dim
    cli_gat_variant = args.gat_variant
    cli_checkpoint_grounding = args.checkpoint_grounding


    # read the given config
    if args.init_checkpoint_config:
        args = load_config(args.init_checkpoint_config)
        args = args[0]

    args.skip_pretrain = skip_pretrain
    args.pretrain_checkpoint = pretrain_checkpoint
    if cli_save_path is not None:
        args.save_path = cli_save_path
    args.eval_every_batches = cli_eval_every_batches
    args.freeze_conf_proj = cli_freeze_conf_proj
    args.attn_entropy_weight = cli_attn_entropy_weight
    args.use_rule_confidence_variant_a = cli_use_rule_confidence_variant_a
    args.attn_dim = cli_attn_dim
    args.gat_variant = cli_gat_variant
    args.checkpoint_grounding = cli_checkpoint_grounding
    if args.save_path is None:
        args.save_path = os.path.join('../outputs', datetime.now().strftime('%Y%m-%d%H-%M%S'))
    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        
    save_config(args)

    set_logger(args.save_path)
    set_seed(args.seed)

    # for grounding dataset
    graph = KnowledgeGraph(args.data_path)
    train_set = TrainDataset(graph, args.g_batch_size)
    valid_set = ValidDataset(graph, args.g_batch_size)
    test_set = TestDataset(graph, args.g_batch_size)
    test_kge_set = TestDataset(graph, 16)
    ruleset = RuleDataset(graph.relation_size, args.rule_file, args.rule_negative_size)

    rules = [rule[0] for rule in ruleset.rules]
    
    
    if args.cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    RulE_model = RulE(graph, args.p_norm, args.mlp_rule_dim, args.gamma_fact, args.gamma_rule, args.hidden_dim, device, args.data_path, attn_dim=args.attn_dim, gat_variant=args.gat_variant, checkpoint_grounding=args.checkpoint_grounding)
    RulE_model.set_rules(rules)

    if not args.skip_pretrain:
        pre_trainer = PreTrainer(
            graph=graph,
            model=RulE_model,
            valid_set=valid_set,
            test_set=test_set,
            ruleset=ruleset,
            expectation=True,
            device = device,
            num_worker=args.cpu_num
        )
        
        pre_trainer.train(args)
        logging.info('Finishing pre-training!')
        del pre_trainer
        torch.cuda.empty_cache()

    print("loading RulE trainer......")

    if args.pretrain_checkpoint:
        checkpoint_path = args.pretrain_checkpoint
    else:
        checkpoint_path = os.path.join(args.save_path, 'checkpoint')

    checkpoint = torch.load(checkpoint_path, map_location=device)
    RulE_model.load_state_dict(checkpoint['model'], strict=False)
    
    logging.info('Loaded pre-trained checkpoint from %s' % checkpoint_path)

    ground_trainer = GroundTrainer(
        model=RulE_model,
        args = args,
        train_set=train_set,
        valid_set=valid_set,
        test_set=test_set,
        test_kge_set = test_kge_set,
        device=device,
        num_worker=args.cpu_num
    )
    
    ground_trainer.train(args)


if __name__ == '__main__':
    
    main()
