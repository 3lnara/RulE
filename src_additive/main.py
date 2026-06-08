
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

    # === Interpretable additive aggregation (replaces the opaque MLP) ===
    parser.add_argument('--simple_aggregation', action='store_true', default=False,
                        help='Replace FuncToNodeSum + score_model (MLP) with a simple '
                             'additive aggregation: score[t] = sum_R w_R * count_R[t] + bias. '
                             'Each rule contributes independently, so contributions are '
                             'directly attributable. Default off -> original MLP.')
    parser.add_argument('--learnable_rule_weight', action='store_true', default=False,
                        help='Lever 1: make the per-rule weight w_R = sigmoid(logit_R) a '
                             'trainable scalar, warm-started from the RulE confidence so init '
                             'matches the frozen baseline. Implies --simple_aggregation. '
                             'Without this flag (but with --simple_aggregation), w_R is frozen '
                             'at the RulE confidence.')

    # === Factorization-Machine pairwise rule interactions ===
    parser.add_argument('--fm_interactions', action='store_true', default=False,
                        help='Add a learnable pairwise FM term over a binary co-fire basis: '
                             'score[t] += sum_{R<R\'} <v_R, v_R\'> * 1[c_R[t]>0] * 1[c_R\'[t]>0]. '
                             'Implies the additive/frozen-linear path; with the linear weights '
                             'frozen, any gain is attributable purely to rule interactions.')
    parser.add_argument('--fm_rank', type=int, default=16,
                        help='Latent dimension k of the per-rule FM embedding v_R.')
    parser.add_argument('--fm_l2', type=float, default=1e-5,
                        help='L2 weight decay applied ONLY to the FM embedding rule_fm_emb '
                             '(main overfitting knob for the interaction term).')
    parser.add_argument('--fm_lr', type=float, default=None,
                        help='Dedicated learning rate for the FM embedding param group. '
                             'The FM embeddings start ~0 (0.01*randn) and must grow before '
                             '<v_R,v_R\'> matters, so they converge slowly at the shared g_lr. '
                             'A higher fm_lr (e.g. 5e-4..1e-3) speeds FM convergence without '
                             'touching the (frozen-linear) bias group. None -> use g_lr.')
    parser.add_argument('--num_iters_override', type=int, default=None,
                        help='If set, overrides num_iters from the JSON config (which otherwise '
                             'wins, since load_config replaces argparse defaults). Lets a single '
                             'run train longer without editing the shared config file.')

    # === NAM (Neural Additive Model) residual shape functions ===
    parser.add_argument('--nam', action='store_true', default=False,
                        help='Add a NAM residual: score[t] += sum_R f_R(count_R[t]) where '
                             'f_R is a shared shape-net conditioned on a small learnable per-rule '
                             'embedding z_R. The shape-net last layer is zero-initialized so the '
                             'run starts at the frozen-linear baseline and learns a nonlinear '
                             'correction. f_R(0)=0 by masking (non-firing rules contribute 0). '
                             'Implies --simple_aggregation.')
    parser.add_argument('--nam_dim', type=int, default=16,
                        help='Dimension of the learnable per-rule embedding z_R used to '
                             'condition the shared shape-net (specialises f_R per rule).')
    parser.add_argument('--nam_hidden', type=int, default=64,
                        help='Hidden width of the shared shape-net MLP.')
    parser.add_argument('--nam_lr', type=float, default=None,
                        help='Dedicated LR for NAM params (nam_net + nam_emb). '
                             'Like FM they start ~0 and need a higher LR to converge. '
                             'None -> fm_lr if set, else g_lr.')
    parser.add_argument('--nam_l2', type=float, default=1e-5,
                        help='Weight decay applied only to NAM params (independent '
                             'regularisation knob for the shape-net).')

    # === Validation-selected alpha sweep (KGE fusion weight) ===
    parser.add_argument('--alpha_sweep', action='store_true', default=False,
                        help='After best-valid reload, sweep --alpha_grid values on the valid '
                             'split (KGE-fused MRR) to pick best_alpha, then report test/test_kge '
                             'at best_alpha. Avoids tuning alpha on the test set.')
    parser.add_argument('--alpha_grid', type=str, default='0,0.5,1,1.5,2,3,4,6,8',
                        help='Comma-separated alpha values to try during --alpha_sweep.')

    # === Early stopping ===
    parser.add_argument('--early_stop_patience', type=int, default=0,
                        help='Stop training if valid MRR has not improved by more than '
                             '--early_stop_min_delta for this many consecutive iterations. '
                             '0 = disabled (train for the full num_iters).')
    parser.add_argument('--early_stop_min_delta', type=float, default=0.0,
                        help='Minimum improvement in valid MRR to count as progress for '
                             'early stopping purposes.')

    return parser.parse_args(args)

def main():
    args = parse_args()

    skip_pretrain = args.skip_pretrain
    pretrain_checkpoint = args.pretrain_checkpoint
    cli_save_path = args.save_path
    cli_simple_aggregation = args.simple_aggregation
    cli_learnable_rule_weight = args.learnable_rule_weight
    cli_fm_interactions = args.fm_interactions
    cli_fm_rank = args.fm_rank
    cli_fm_l2 = args.fm_l2
    cli_fm_lr = args.fm_lr
    cli_num_iters_override = args.num_iters_override
    cli_nam = args.nam
    cli_nam_dim = args.nam_dim
    cli_nam_hidden = args.nam_hidden
    cli_nam_lr = args.nam_lr
    cli_nam_l2 = args.nam_l2
    cli_alpha_sweep = args.alpha_sweep
    cli_alpha_grid = args.alpha_grid
    cli_early_stop_patience = args.early_stop_patience
    cli_early_stop_min_delta = args.early_stop_min_delta

    # read the given config
    if args.init_checkpoint_config:
        args = load_config(args.init_checkpoint_config)
        args = args[0]

    args.skip_pretrain = skip_pretrain
    args.pretrain_checkpoint = pretrain_checkpoint
    if cli_save_path is not None:
        args.save_path = cli_save_path
    args.simple_aggregation = cli_simple_aggregation
    args.learnable_rule_weight = cli_learnable_rule_weight
    args.fm_interactions = cli_fm_interactions
    args.fm_rank = cli_fm_rank
    args.fm_l2 = cli_fm_l2
    args.fm_lr = cli_fm_lr
    if cli_num_iters_override is not None:
        args.num_iters = cli_num_iters_override
    args.nam = cli_nam
    args.nam_dim = cli_nam_dim
    args.nam_hidden = cli_nam_hidden
    args.nam_lr = cli_nam_lr
    args.nam_l2 = cli_nam_l2
    args.alpha_sweep = cli_alpha_sweep
    args.alpha_grid = cli_alpha_grid
    args.early_stop_patience = cli_early_stop_patience
    args.early_stop_min_delta = cli_early_stop_min_delta

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
    
    # checkpoint = torch.load(os.path.join(args.save_path, 'checkpoint'))
    # RulE_model.load_state_dict(checkpoint['model'])


    # valid_mrr = pre_trainer.evaluate('valid', expectation=True)
    # test_mrr = pre_trainer.evaluate('test', expectation=True)
    
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

    # logging.info('Test the results of pre-training')
    
    # valid_mrr = pre_trainer.evaluate('valid', expectation=True)
    # test_mrr = pre_trainer.evaluate('test', expectation=True)

    # RulE_model.add_param()

    # checkpoint = torch.load(os.path.join(args.save_path, 'grounding.pt'))
    # RulE_model.load_state_dict(checkpoint['model'])

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

    # valid_mrr = ground_trainer.evaluate('valid', expectation=True)
    # test_mrr = ground_trainer.evaluate('test', expectation=True)
    
    # args.g_batch_size = 32
    
    ground_trainer.train(args)
    
    # return test_mrr


if __name__ == '__main__':
    
    main()
