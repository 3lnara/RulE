
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

    # === Eval-only rank dump (reuse a trained grounding.pt, no retraining) ===
    parser.add_argument('--eval_only', action='store_true', default=False,
                        help='Skip grounding training; load --grounding_checkpoint and '
                             'evaluate valid/test only. Implies --skip_pretrain.')
    parser.add_argument('--grounding_checkpoint', type=str, default=None,
                        help='Path to a trained grounding.pt to load for --eval_only.')
    parser.add_argument('--dump_ranks', action='store_true', default=False,
                        help='Write per-query filtered ranks (h, r, t, L, H) to '
                             'ranks_mlp_<split>.csv in --save_path during evaluation.')
    return parser.parse_args(args)

def main():
    args = parse_args()

    skip_pretrain = args.skip_pretrain
    pretrain_checkpoint = args.pretrain_checkpoint
    cli_save_path = args.save_path
    cli_eval_only = args.eval_only
    cli_grounding_checkpoint = args.grounding_checkpoint
    cli_dump_ranks = args.dump_ranks

    # read the given config
    if args.init_checkpoint_config:
        args = load_config(args.init_checkpoint_config)
        args = args[0]

    args.skip_pretrain = skip_pretrain
    args.pretrain_checkpoint = pretrain_checkpoint
    args.eval_only = cli_eval_only
    args.grounding_checkpoint = cli_grounding_checkpoint
    args.dump_ranks = cli_dump_ranks
    # --eval_only loads a trained grounding.pt and never pre-trains.
    if cli_eval_only:
        args.skip_pretrain = True
    if cli_save_path is not None:
        args.save_path = cli_save_path

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

    # In --eval_only mode the full trained grounding.pt (which already contains
    # KGE/rule embeddings + the MLP head) is loaded inside eval_and_dump, so the
    # pre-train checkpoint load is skipped here.
    if not getattr(args, 'eval_only', False):
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

    if getattr(args, 'eval_only', False):
        ground_trainer.eval_and_dump(args)
    else:
        ground_trainer.train(args)
    
    # return test_mrr


if __name__ == '__main__':
    
    main()
