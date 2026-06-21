
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

    parser.add_argument('--seed', default=-1, type=int,
                        help='Random seed. -1 (default) means use the value from the JSON config.')
    
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
    parser.add_argument('--num_iters', default=None, type=int)

    parser.add_argument('--skip_pretrain', action='store_true', default=False,
                        help='Skip pre-training and load from an existing checkpoint')
    parser.add_argument('--pretrain_checkpoint', type=str, default=None,
                        help='Path to pre-trained checkpoint (used with --skip_pretrain)')

    # === Interpretable additive aggregation (replaces the opaque MLP) ===
    parser.add_argument('--simple_aggregation', action='store_true', default=False,
                        help='Replace FuncToNodeSum + score_model (MLP) with a simple '
                             'additive aggregation: score[t] = sum_R w_R * count_R[t] + bias. '
                             'Each rule contributes independently, so contributions are '
                             'directly attributable. Default off -> original MLP.')

    # === Faithful paper "sum (w/o MLP)" ===
    parser.add_argument('--paper_sum', action='store_true', default=False,
                        help="Faithful reproduction of the paper's 'sum (w/o MLP)' baseline: "
                             "binary activation (1 if rule fired, 0 otherwise), no per-entity "
                             "bias, raw frozen RulE confidence as weights. Implies "
                             "--simple_aggregation.")
    parser.add_argument('--no_bias', action='store_true', default=False,
                        help='Drop the per-entity bias from --simple_aggregation so the '
                             'score is purely Sum_R w_R * count_R[t] (raw counts, no '
                             'popularity prior). Implies --simple_aggregation. The bias is '
                             'frozen, so this mode has no trainable parameters (fixed scorer).')
    parser.add_argument('--clamp_negative_confidence', action='store_true', default=False,
                        help='Clamp frozen per-rule RulE confidences to >=0 '
                             '(w_R = max(0, w_R)) before additive aggregation, zeroing out '
                             'negative-confidence rules. Only affects '
                             '--simple_aggregation / --paper_sum runs.')

    return parser.parse_args(args)

def main():
    args = parse_args()

    skip_pretrain = args.skip_pretrain
    pretrain_checkpoint = args.pretrain_checkpoint
    cli_save_path = args.save_path
    cli_simple_aggregation = args.simple_aggregation
    cli_paper_sum = args.paper_sum
    cli_no_bias = args.no_bias
    cli_clamp_negative_confidence = args.clamp_negative_confidence
    cli_num_iters = args.num_iters  # None-sentinel below: argparse default is 20
    # Sentinel: argparse default is -1 meaning "not provided, use config value".
    cli_seed = args.seed if args.seed != -1 else None

    # read the given config
    if args.init_checkpoint_config:
        args = load_config(args.init_checkpoint_config)
        args = args[0]

    args.skip_pretrain = skip_pretrain
    args.pretrain_checkpoint = pretrain_checkpoint
    if cli_save_path is not None:
        args.save_path = cli_save_path
    args.paper_sum = cli_paper_sum
    args.no_bias = cli_no_bias
    args.clamp_negative_confidence = cli_clamp_negative_confidence
    # --paper_sum and --no_bias both force simple_aggregation
    if cli_paper_sum or cli_no_bias:
        args.simple_aggregation = True
    else:
        args.simple_aggregation = cli_simple_aggregation
    # CLI --num_iters overrides the JSON config value when explicitly provided.
    if cli_num_iters is not None:
        args.num_iters = cli_num_iters
    # CLI --seed overrides the JSON config value when explicitly provided.
    if cli_seed is not None:
        args.seed = cli_seed

    # wandb.init(project='RulE',group='RotatE', name = args.save_path, config=args)
    if args.save_path is None:
        args.save_path = os.path.join('../outputs', datetime.now().strftime('%Y%m-%d%H-%M%S'))
    # else:
    #     args.save_path = '../outputs/'+ args.save_path
    
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

    RulE_model = RulE(graph, args.p_norm, args.mlp_rule_dim, args.gamma_fact, args.gamma_rule, args.hidden_dim, device, args.data_path)
    RulE_model.set_rules(rules)

    
    # For pre-training 
    if not args.skip_pretrain:
        pre_trainer = PreTrainer(
            graph=graph,
            model=RulE_model,
            valid_set=valid_set,
            test_set=test_set,
            # tripletset=kge_train_set,
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

    # load rule embedding and KGE embedding

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
