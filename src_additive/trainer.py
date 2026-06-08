
from utils import *
import torch
from torch import nn

from torch.utils.data import DataLoader
from itertools import islice
from data import Iterator, RuleDataset, KGETrainDataset, BidirectionalOneShotIterator
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from layers import MLP

class PreTrainer(object):

    def __init__(self, graph, model, valid_set, test_set, ruleset, expectation, device, num_worker=0):
        
        
        self.num_worker = num_worker
        self.device = device
      
       
        if self.device.type == "cuda":
            model = model.cuda(self.device)

        self.graph = graph
        self.model = model
        self.valid_set = valid_set
        self.test_set = test_set
        
        self.RuleSet = ruleset
        self.expectation = expectation
        

    
    def train(self, args):
        
        # Set training configuration
       
        current_learning_rate = float(args.learning_rate)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=float(args.learning_rate), 
            weight_decay=float(args.weight_decay)
            )
            
        
        triplets_dataloader_head = DataLoader(
            KGETrainDataset(
                triples=self.graph.train_facts,
                nentity=self.graph.entity_size, 
                nrelation=self.graph.relation_size,
                negative_sample_size=args.negative_sample_size,
                mode = 'head-batch'),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=KGETrainDataset.collate_fn
        )

        triplets_dataloader_tail = DataLoader(
            KGETrainDataset(
                triples=self.graph.train_facts,
                nentity=self.graph.entity_size, 
                nrelation=self.graph.relation_size,
                negative_sample_size=args.negative_sample_size,
                mode = 'tail-batch'),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=KGETrainDataset.collate_fn
        )


        rules_dataloader = DataLoader(
            self.RuleSet, 
            batch_size=args.rule_batch_size, 
            shuffle=True, 
            num_workers=max(1, args.cpu_num//2),
            collate_fn=RuleDataset.collate_fn)

        self.triplets_iterator = BidirectionalOneShotIterator(triplets_dataloader_head, triplets_dataloader_tail)
        self.rules_iterator = Iterator(rules_dataloader)

        logging.info('>>>>> ruleE: Pre-training')
        training_logs = []
        best_mrr = 0.0

        if args.warm_up_steps:
            warm_up_steps = args.warm_up_steps
        else:
            warm_up_steps = args.max_steps // 2

        for step in range(0, args.max_steps + 1):

            log = self.train_step( optimizer, self.triplets_iterator, self.rules_iterator, args)
            
            training_logs.append(log)

            if step >= warm_up_steps:
                current_learning_rate = current_learning_rate / 10
                logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, self.model.parameters()), 
                    lr=current_learning_rate
                )
                warm_up_steps = warm_up_steps * 3
                if args.disable_adv:
                    args.adversarial_temperature = 0
            
            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
                log_metrics('Training average', step, metrics)
                training_logs = []
                
            if step % args.valid_steps == 0:
                logging.info('Evaluating on Valid Dataset...')
                torch.cuda.empty_cache()
                mrr = self.evaluate("valid", self.expectation )
                if mrr > best_mrr:
                    save_model(self.model,optimizer, args)
                    best_mrr = mrr
                # save_model(self.model,optimizer, args)


    def train_step(self, optimizer, triplets_iterator, rules_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''
        
        # self.model.rule_emb.requires_grad = False
        # # self.model.rule_emb.weight.data = self.model.rule_emb.weight / self.model.rule_emb.weight.norm(dim=-1,keepdim=True)
        # self.model.rule_emb.weight.data  = F.normalize( self.model.rule_emb.weight.data , p=2, dim=-1)
        # self.model.rule_emb.requires_grad = True

        model = self.model
        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, mode= next(triplets_iterator)
        positive_rule, negative_idx, negative_rule, mode_rule, rule_mask = next(rules_iterator)

        if self.device.type == "cuda":
            positive_sample = positive_sample.cuda(self.device)
            negative_sample = negative_sample.cuda(self.device)
            positive_rule = positive_rule.cuda(self.device)
            negative_idx = negative_idx.cuda(self.device)
            negative_rule = negative_rule.cuda(self.device)
            rule_mask = rule_mask.cuda(self.device)
            subsampling_weight = subsampling_weight.cuda(self.device)

        negative_fact_score, _ = model.compute_KGE((positive_sample, negative_sample), mode) 
        negative_rule_score = model.compute_ruleE((positive_rule,  rule_mask, negative_idx, negative_rule), mode=mode_rule) 
        # print(args.adversarial_temperature)
        negative_fact_score = (F.softmax(negative_fact_score * args.adversarial_temperature, dim = 1).detach() 
                            * F.logsigmoid(-negative_fact_score)).sum(dim = 1)
        
        negative_rule_score = (F.softmax(negative_rule_score * args.adversarial_temperature, dim = 1).detach() 
                            * F.logsigmoid(-negative_rule_score)).sum(dim = 1)
        
        
        # negative_rule_score = F.logsigmoid(-negative_rule_score).mean(dim = 1)


        positive_fact_score, ent = model.compute_KGE(positive_sample)
        positive_rule_score = model.compute_ruleE((positive_rule,rule_mask))


        positive_fact_score = F.logsigmoid(positive_fact_score).squeeze(dim = 1)
        positive_rule_score = F.logsigmoid(positive_rule_score)

        negative_rule_score_weight = negative_rule_score 
        positive_rule_score_weight = positive_rule_score 

        if args.uni_weight:
            positive_fact_loss = - positive_fact_score.mean() 
            negative_fact_loss = - negative_fact_score.mean()
        else:
            positive_fact_loss = - (subsampling_weight * positive_fact_score).sum()/subsampling_weight.sum()            
            negative_fact_loss = - (subsampling_weight * negative_fact_score).sum()/subsampling_weight.sum() 

        
        positive_rule_loss = - positive_rule_score_weight.mean() * args.weight_rule
        negative_rule_loss = - negative_rule_score_weight.mean() * args.weight_rule


        loss_fact = (positive_fact_loss + negative_fact_loss)/2
        loss_rule = (positive_rule_loss + negative_rule_loss)/2

        loss = loss_rule + loss_fact
        # loss = loss_fact
        # loss = loss_rule

        # wandb.log({'train/loss_fact':loss_fact, 'train/loss_rule':loss_rule})
        # wandb.log({ 'train/loss_rule':loss_rule})
        if args.regularization:
            # Use regularization
            regularization = args.regularization * (
                ent[0].norm(p=2)**2 +
                ent[1].norm(p=2)**2
            ) / ent[0].shape[0]
            # regularization_rule = args.regularization * (
            #     rule.norm(p=2) ** 2
            # ) / rule[0].shape[0]
            loss = loss + regularization 
        else:
            regularization = torch.tensor([0])

        loss.backward()

        optimizer.step()

        log = {
            
            'positive_fact_loss': positive_fact_loss.item(),
            'negative_fact_loss': negative_fact_loss.item(),
            'positive_rule_loss': positive_rule_loss.item(),
            'negative_rule_loss': negative_rule_loss.item(),
            'regularization': regularization.item(),
            'loss': loss.item()
        }

        return log

    @torch.no_grad()
    def evaluate(self, split, expectation=True):
        
        logging.info('>>>>> RuleE emb: Evaluating on {}'.format(split))
        
        test_set = getattr(self, "%s_set" % split)
        # test_set = self.test_set_data
        dataloader = DataLoader(test_set, batch_size=1, num_workers=self.num_worker)
        model = self.model

        model.eval()
        concat_logits = []
        concat_all_h = []
        concat_all_r = []
        concat_all_t = []
        concat_flag = []
        # concat_mask = []
        for batch in dataloader:
            all_h, all_r, all_t, flag = batch
            all_h = all_h.squeeze(0)
            all_r = all_r.squeeze(0)
            all_t = all_t.squeeze(0)
            flag = flag.squeeze(0)
            if self.device.type == "cuda":
                all_h = all_h.cuda(device=self.device)
                all_r = all_r.cuda(device=self.device)
                all_t = all_t.cuda(device=self.device)
                flag = flag.cuda(device=self.device)

            # Process one query at a time to avoid OOM.
            # compute_g_KGE with B queries allocates [B, 40943, 1000] = B * 164MB.
            # With B=g_batch_size=32 that is 5.24GB; B=1 keeps it at 164MB.
            kge_rows = []
            for i in range(all_h.size(0)):
                kge_rows.append(model.compute_g_KGE(all_h[i:i+1], all_r[i:i+1]))
            KGE_score = torch.cat(kge_rows, dim=0)

            logits = KGE_score

            concat_logits.append(logits)
            concat_all_h.append(all_h)
            concat_all_r.append(all_r)
            concat_all_t.append(all_t)
            concat_flag.append(flag)

        
        concat_logits = torch.cat(concat_logits, dim=0)
        concat_all_h = torch.cat(concat_all_h, dim=0)
        concat_all_r = torch.cat(concat_all_r, dim=0)
        concat_all_t = torch.cat(concat_all_t, dim=0)
        concat_flag = torch.cat(concat_flag, dim=0)

        
        ranks = []
        for k in range(concat_all_t.size(0)):
            h = concat_all_h[k]
            r = concat_all_r[k]
            t = concat_all_t[k]
            val = concat_logits[k, t]

            L = (concat_logits[k][concat_flag[k]] > val).sum().item() + 1
            H = (concat_logits[k][concat_flag[k]] >= val).sum().item() + 2
            ranks += [[h, r, t, L, H]]
        ranks = torch.tensor(ranks, dtype=torch.long, device=self.device)
            
        query2LH = dict()
        for h, r, t, L, H in ranks.data.cpu().numpy().tolist():
            query2LH[(h, r, t)] = (L, H)
            
        hit1, hit3, hit10, mr, mrr = 0.0, 0.0, 0.0, 0.0, 0.0
        for (L, H) in query2LH.values():
            if expectation:
                for rank in range(L, H):
                    if rank <= 1:
                        hit1 += 1.0 / (H - L)
                    if rank <= 3:
                        hit3 += 1.0 / (H - L)
                    if rank <= 10:
                        hit10 += 1.0 / (H - L)
                    mr += rank / (H - L)
                    mrr += 1.0 / rank / (H - L)
            else:
                rank = H - 1
                if rank <= 1:
                    hit1 += 1
                if rank <= 3:
                    hit3 += 1
                if rank <= 10:
                    hit10 += 1
                mr += rank
                mrr += 1.0 / rank
        
        hit1 /= len(ranks)
        hit3 /= len(ranks)
        hit10 /= len(ranks)
        mr /= len(ranks)
        mrr /= len(ranks)

        
        logging.info('Data : {}'.format(len(query2LH)))
        logging.info('Hit1 : {:.6f}'.format(hit1))
        logging.info('Hit3 : {:.6f}'.format(hit3))
        logging.info('Hit10: {:.6f}'.format(hit10))
        logging.info('MR   : {:.6f}'.format(mr))
        logging.info('MRR  : {:.6f}'.format(mrr))

        return mrr

    def load(self, checkpoint, load_optimizer=True):
        """
        Load a checkpoint from file.
        Parameters:
            checkpoint (file-like): checkpoint file
            load_optimizer (bool, optional): load optimizer state or not
        """
        
        logging.info("Load checkpoint from %s" % checkpoint)
        checkpoint = os.path.expanduser(checkpoint)
        state = torch.load(checkpoint, map_location=self.device)

        self.model.load_state_dict(state["model"])

        if load_optimizer:
            self.optimizer.load_state_dict(state["optimizer"])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)


    def save(self, checkpoint):
        """
        Save checkpoint to file.
        Parameters:
            checkpoint (file-like): checkpoint file
        """
       
        logging.info("Save checkpoint to %s" % checkpoint)
        checkpoint = os.path.expanduser(checkpoint)
        if self.rank == 0:
            state = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict()
            }
            torch.save(state, checkpoint)




class GroundTrainer(object):
    
    def __init__(self, model, args, train_set, valid_set, test_set,test_kge_set, device, num_worker=0):
        
        self.num_worker = num_worker

        
        self.device = device
        if self.device.type == "cuda":
            model = model.cuda(self.device)

        self.args = args
        self.model = model
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set
        self.test_kge_set = test_kge_set
        
    def train(self, args):
        
        # fix the parameters of pre-training

        self.model.entity_embedding.weight.requires_grad = False
        self.model.relation_embedding.weight.requires_grad = False
        self.model.rule_emb.weight.requires_grad = False

        # === Interpretable additive aggregation: per-rule weight setup ===
        # Active when --simple_aggregation or --learnable_rule_weight is set. The
        # per-rule weight w_R = sigmoid(logit_R) is warm-started from the RulE
        # confidence (sigmoid(add_ruleE)), so at init the additive model matches the
        # frozen baseline. add_ruleE depends only on the (frozen) relation/rule
        # embeddings, so this is independent of eval_compute_rule_weight below.
        # Created BEFORE the optimizer so a learnable weight joins the param group.
        use_additive = (getattr(args, 'simple_aggregation', False)
                        or getattr(args, 'learnable_rule_weight', False)
                        or getattr(args, 'fm_interactions', False)
                        or getattr(args, 'nam', False))
        if use_additive:
            rule_features = self.model.rule_features.to(self.device)
            rule_masks = self.model.rule_masks.to(self.device)
            batch = 128
            split_num = max(1, rule_features.size(0) // batch)
            init_logits = []
            with torch.no_grad():
                for rules, masks in zip(
                    torch.split(rule_features, split_num, 0),
                    torch.split(rule_masks, split_num, 0),
                ):
                    scores, _ = self.model.add_ruleE(rules.unsqueeze(1), masks)
                    init_logits.append(scores.squeeze(1))
            init_logits = torch.cat(init_logits, dim=0).detach()
            if getattr(args, 'learnable_rule_weight', False):
                self.model.rule_weight_logit = nn.Parameter(init_logits.clone())
                logging.info('[lever1] learnable per-rule weight ENABLED; warm-started '
                             'from RulE confidence (num_rules=%d).' % init_logits.numel())
            else:
                self.model.register_buffer('rule_weight_logit', init_logits.clone())
                logging.info('[additive] simple aggregation with FROZEN per-rule weight '
                             '(RulE confidence; num_rules=%d).' % init_logits.numel())
            self.model.simple_aggregation = True

        # === Pairwise FM interactions: per-rule embedding setup ===
        # One latent vector v_R per rule; the pairwise term is built in
        # model.forward via the binary co-fire squared-sum identity. Created
        # BEFORE the optimizer so it joins the parameter groups. Initialised to
        # small noise (NOT zero): at v=0 the FM gradient vanishes (saddle), so
        # zero-init would never learn; 0.01*randn starts FM ~= 0 yet trainable,
        # keeping the model ~= the frozen-linear baseline at init.
        if getattr(args, 'fm_interactions', False):
            num_rules = self.model.rule_features.size(0)
            k = int(getattr(args, 'fm_rank', 16))
            self.model.rule_fm_emb = nn.Parameter(
                0.01 * torch.randn(num_rules, k, device=self.device))
            self.model.use_fm = True
            logging.info('[fm] FM interactions ENABLED (rank=%d, l2=%g, num_rules=%d).'
                         % (k, float(getattr(args, 'fm_l2', 1e-5)), num_rules))

        # === NAM residual: shared shape-net + per-rule embedding ===
        # Created BEFORE the optimizer so it joins the parameter groups.
        # The shape-net final layer is zero-initialized so the NAM residual is
        # exactly 0 at iter-0, making the starting point identical to the
        # frozen-linear baseline. nam_emb uses small-noise init (same rationale
        # as FM: zero would stall gradients at the saddle).
        if getattr(args, 'nam', False):
            num_rules = self.model.rule_features.size(0)
            nam_dim = int(getattr(args, 'nam_dim', 16))
            nam_hidden = int(getattr(args, 'nam_hidden', 64))
            self.model.nam_emb = nn.Parameter(
                0.01 * torch.randn(num_rules, nam_dim, device=self.device))
            self.model.nam_net = MLP(2 + nam_dim, [nam_hidden, 1]).to(self.device)
            nn.init.zeros_(self.model.nam_net.layers[-1].weight)
            nn.init.zeros_(self.model.nam_net.layers[-1].bias)
            self.model.use_nam = True
            logging.info('[nam] NAM residual ENABLED (nam_dim=%d, hidden=%d, '
                         'l2=%g, num_rules=%d).'
                         % (nam_dim, nam_hidden,
                            float(getattr(args, 'nam_l2', 1e-5)), num_rules))

        # Build the optimizer with per-component param groups so that FM and NAM
        # can use a higher dedicated LR (both start ~0 and converge slowly at
        # the shared g_lr) and independent L2 regularisation.
        #
        # Groups constructed dynamically: base (bias + any other requires_grad
        # params not claimed by FM/NAM), fm (rule_fm_emb), nam (nam_net + nam_emb).
        # The set-of-ids trick deduplicates across groups.
        claimed_ids: set = set()
        param_groups = []

        if getattr(args, 'fm_interactions', False):
            fm_params = [self.model.rule_fm_emb]
            for p in fm_params:
                claimed_ids.add(id(p))
            fm_lr = getattr(args, 'fm_lr', None)
            fm_lr = float(fm_lr) if fm_lr is not None else float(args.g_lr)
            param_groups.append({
                'params': fm_params,
                'lr': fm_lr,
                'weight_decay': float(getattr(args, 'fm_l2', 1e-5)),
            })

        if getattr(args, 'nam', False):
            nam_params = (list(self.model.nam_net.parameters())
                          + [self.model.nam_emb])
            for p in nam_params:
                claimed_ids.add(id(p))
            fm_lr_fallback = getattr(args, 'fm_lr', None)
            fm_lr_fallback = float(fm_lr_fallback) if fm_lr_fallback is not None else float(args.g_lr)
            nam_lr = getattr(args, 'nam_lr', None)
            nam_lr = float(nam_lr) if nam_lr is not None else fm_lr_fallback
            param_groups.append({
                'params': nam_params,
                'lr': nam_lr,
                'weight_decay': float(getattr(args, 'nam_l2', 1e-5)),
            })

        base_params = [p for p in self.model.parameters()
                       if p.requires_grad and id(p) not in claimed_ids]
        param_groups.insert(0, {
            'params': base_params,
            'lr': float(args.g_lr),
            'weight_decay': float(args.weight_decay),
        })

        optimizer = torch.optim.Adam(param_groups, lr=float(args.g_lr))

        # Log the final optimizer layout for reproducibility.
        use_fm = getattr(args, 'fm_interactions', False)
        use_nam = getattr(args, 'nam', False)
        fm_lr_log = float(args.fm_lr) if getattr(args, 'fm_lr', None) is not None else float(args.g_lr)
        nam_lr_fallback = getattr(args, 'fm_lr', None)
        nam_lr_fallback = float(nam_lr_fallback) if nam_lr_fallback is not None else float(args.g_lr)
        # nam_lr_log = float(args.nam_lr) if getattr(args, 'nam_lr', None) is not None else float(args.nam_lr_fallback)
        logging.info('[optim] base_lr=%g%s%s'
                     % (float(args.g_lr),
                        (', fm_lr=%g fm_l2=%g' % (fm_lr_log, float(getattr(args, 'fm_l2', 1e-5))))
                        if use_fm else ''))
                        # (', nam_lr=%g nam_l2=%g' % (nam_lr_log, float(getattr(args, 'nam_l2', 1e-5))))
                        # if use_nam else ''))


        self.train_set.make_batches()
        
        train_dataloader = DataLoader(self.train_set, 1, num_workers=self.num_worker)
        
        self.model.eval_compute_rule_weight(self.device)


        
        logging.info('>>>>> RulE: Grounding-Training')
        

        best_valid_mrr = 0.0 
        test_mrr = 0.0

        # Early stopping state. patience=0 means disabled.
        es_patience = int(getattr(args, 'early_stop_patience', 0))
        es_min_delta = float(getattr(args, 'early_stop_min_delta', 0.0))
        iters_no_improve = 0

        warm_up_steps = args.num_iters // 2
        current_learning_rate = float(args.g_lr)

        for k in range(args.num_iters):

            
            logging.info('-------------------------')
            logging.info('| Iteration: {}/{}'.format(k + 1, args.num_iters))
            logging.info('-------------------------')
        
            # if k >= warm_up_steps:

            #     current_learning_rate = current_learning_rate / 10
            #     logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, k))
            #     optim = torch.optim.Adam(
            #         filter(lambda p: p.requires_grad, predictor.parameters()), 
            #         lr=current_learning_rate
            #     )
            #     warm_up_steps = warm_up_steps * 3

            self.train_step( optimizer, train_dataloader, args.batch_per_epoch, args.smoothing, args.print_every, args)
            valid_mrr_iter = self.evaluate('valid', args.alpha, expectation=True)

            if valid_mrr_iter > best_valid_mrr + es_min_delta:
                best_valid_mrr = valid_mrr_iter
                test_mrr = valid_mrr_iter
                self.save(args, os.path.join(args.save_path, 'grounding.pt'))
                iters_no_improve = 0
            else:
                iters_no_improve += 1
                if es_patience > 0 and iters_no_improve >= es_patience:
                    logging.info('[early_stop] No improvement for %d iterations '
                                 '(patience=%d, min_delta=%g). Stopping.'
                                 % (iters_no_improve, es_patience, es_min_delta))
                    break
        

        logging.info('-------------------------')
        logging.info('| Best Valid MRR (model selection): {:.6f}'.format(test_mrr))
        logging.info('-------------------------')

        # Reload the best-valid checkpoint so the final valid/test/test_kge
        # numbers below reflect the selected model, not the last training state.
        checkpoint = torch.load(os.path.join(self.args.save_path, 'grounding.pt'))
        self.model.load_state_dict(checkpoint['model'])
        
        test_mrr_iter = self.evaluate('valid', args.alpha, expectation=True)
        test_mrr_iter = self.evaluate('test', args.alpha, expectation=True)
        # Use evaluate_t_with_sweep so that if --alpha_sweep is set, the best
        # alpha is selected on valid before the final test_kge report.
        test_mrr_iter = self.evaluate_t_with_sweep('test_kge', args, expectation=True)


       
       

    def train_step(self, optimizer, train_dataloader, batch_per_epoch, smoothing, print_every, args):
        
        batch_per_epoch = batch_per_epoch or len(train_dataloader)
        model = self.model
        
        model.train()

        total_loss = 0.0
        total_size = 0.0

        MEM_LOG_BATCHES = 3   # log memory for first N batches only

        for batch_id, batch in enumerate(islice(train_dataloader, batch_per_epoch)):
            import model as _model_mod
            _model_mod._MEM_LOG = (batch_id < MEM_LOG_BATCHES)
            if batch_id < MEM_LOG_BATCHES and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                logging.info(f"[MEM] ===== batch {batch_id} peak reset =====")

            all_h, all_r, all_t, target, edges_to_remove = batch
            all_h = all_h.squeeze(0)
            all_r = all_r.squeeze(0)
            all_t = all_t.squeeze(0)
            target = target.squeeze(0)
            edges_to_remove = edges_to_remove.squeeze(0)
            target_t = torch.nn.functional.one_hot(all_t, self.train_set.graph.entity_size)
            
            if self.device.type == "cuda":
                all_h = all_h.cuda(device=self.device)
                all_r = all_r.cuda(device=self.device)
                all_t = all_t.cuda(device=self.device)
                target = target.cuda(device=self.device)
                edges_to_remove = edges_to_remove.cuda(device=self.device)
                target_t = target_t.cuda(device=self.device)

            target = target * smoothing + target_t * (1 - smoothing)
            
            grounding_rule_score, mask = model(all_h, all_r, edges_to_remove)
            
            if mask.sum().item() != 0:
                rule_logits = (torch.softmax(grounding_rule_score, dim=1) + 1e-8).log()
                
                loss = -(rule_logits[mask] * target[mask]).sum() / torch.clamp(target[mask].sum(), min=1)
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()
                total_size += mask.sum().item()
            
            if (batch_id + 1) % print_every == 0:
                logging.info('loss:    {} {} {:.6f} {:.1f}'.format(batch_id + 1, len(train_dataloader), loss, total_size / print_every))
                
                total_loss = 0.0
                total_size = 0.0
                # Intentionally NOT checkpointing here: grounding.pt is written
                # only on a best-valid improvement in train(), so the model
                # reloaded for the final test/test_kge eval is the best-valid one
                # rather than whatever mid-epoch state was saved last.
        

    @torch.no_grad()
    def evaluate(self, split, alpha=3.0, expectation=True):

        logging.info('>>>>> Predictor: Evaluating on {}'.format(split))
        test_set = getattr(self, "%s_set" % split)
        
        dataloader = DataLoader(test_set, 1, num_workers=self.num_worker)
        model = self.model

        model.eval()
        concat_logits = []
        concat_all_h = []
        concat_all_r = []
        concat_all_t = []
        concat_flag = []
        concat_mask = []
        
        for batch in tqdm(dataloader):

            all_h, all_r, all_t, flag = batch
            all_h = all_h.squeeze(0)
            all_r = all_r.squeeze(0)
            all_t = all_t.squeeze(0)
            flag = flag.squeeze(0)

            if self.device.type == "cuda":
                all_h = all_h.cuda(device=self.device)
                all_r = all_r.cuda(device=self.device)
                all_t = all_t.cuda(device=self.device)
                flag = flag.cuda(device=self.device)

            # logits, mask = model.forward_weight(all_h, all_r, None)
            logits, mask = model(all_h, all_r, None)

            # kge_score = model.compute_g_KGE(all_h,all_r)
            # logits += alpha * kge_score

            concat_logits.append(logits)
            concat_all_h.append(all_h)
            concat_all_r.append(all_r)
            concat_all_t.append(all_t)
            concat_flag.append(flag)
            concat_mask.append(mask)
        
        concat_logits = torch.cat(concat_logits, dim=0)
        concat_all_h = torch.cat(concat_all_h, dim=0)
        concat_all_r = torch.cat(concat_all_r, dim=0)
        concat_all_t = torch.cat(concat_all_t, dim=0)
        concat_flag = torch.cat(concat_flag, dim=0)
        concat_mask = torch.cat(concat_mask, dim=0)
        
        ranks = []
        for k in range(concat_all_t.size(0)):
            h = concat_all_h[k]
            r = concat_all_r[k]
            t = concat_all_t[k]
            if concat_mask[k, t].item() == True:
                val = concat_logits[k, t]
                L = (concat_logits[k][concat_flag[k]] > val).sum().item() + 1
                H = (concat_logits[k][concat_flag[k]] >= val).sum().item() + 2
            else:
                L = 1
                H = test_set.graph.entity_size + 1
            ranks += [[h, r, t, L, H]]
        ranks = torch.tensor(ranks, dtype=torch.long, device=self.device)
            
        query2LH = dict()
        for h, r, t, L, H in ranks.data.cpu().numpy().tolist():
            query2LH[(h, r, t)] = (L, H)
            
        hit1, hit3, hit10, mr, mrr = 0.0, 0.0, 0.0, 0.0, 0.0
        for (L, H) in query2LH.values():
            if expectation:
                for rank in range(L, H):
                    if rank <= 1:
                        hit1 += 1.0 / (H - L)
                    if rank <= 3:
                        hit3 += 1.0 / (H - L)
                    if rank <= 10:
                        hit10 += 1.0 / (H - L)
                    mr += rank / (H - L)
                    mrr += 1.0 / rank / (H - L)
            else:
                rank = H - 1
                if rank <= 1:
                    hit1 += 1
                if rank <= 3:
                    hit3 += 1
                if rank <= 10:
                    hit10 += 1
                mr += rank
                mrr += 1.0 / rank
            
        hit1 /= len(ranks)
        hit3 /= len(ranks)
        hit10 /= len(ranks)
        mr /= len(ranks)
        mrr /= len(ranks)

        
        logging.info('Data : {}'.format(len(query2LH)))
        logging.info('Hit1 : {:.6f}'.format(hit1))
        logging.info('Hit3 : {:.6f}'.format(hit3))
        logging.info('Hit10: {:.6f}'.format(hit10))
        logging.info('MR   : {:.6f}'.format(mr))
        logging.info('MRR  : {:.6f}'.format(mrr))
    
        
        return mrr


    @torch.no_grad()
    def _collect_kge_scores(self, split):
        """Collect per-query (rule_logits, kge_score, flag, mask, all_h, all_r, all_t)
        for the KGE-fused evaluation path, without applying alpha.

        Returns a dict of tensors that can be combined with an arbitrary alpha via
        _mrr_from_logits(), allowing a cheap alpha sweep without re-running forward.
        """
        test_set = getattr(self, "%s_set" % split)
        dataloader = DataLoader(test_set, 1, num_workers=self.num_worker)
        model = self.model
        model.eval()

        rule_logits_list, kge_list, all_h_list, all_r_list, all_t_list, flag_list, mask_list = \
            [], [], [], [], [], [], []

        for batch in tqdm(dataloader):
            all_h, all_r, all_t, flag = batch
            all_h = all_h.squeeze(0)
            all_r = all_r.squeeze(0)
            all_t = all_t.squeeze(0)
            flag  = flag.squeeze(0)

            if self.device.type == "cuda":
                all_h = all_h.cuda(device=self.device)
                all_r = all_r.cuda(device=self.device)
                all_t = all_t.cuda(device=self.device)
                flag  = flag.cuda(device=self.device)

            logits, mask = model(all_h, all_r, None)
            kge_score = model.compute_g_KGE(all_h, all_r)

            rule_logits_list.append(logits)
            kge_list.append(kge_score)
            all_h_list.append(all_h)
            all_r_list.append(all_r)
            all_t_list.append(all_t)
            flag_list.append(flag)
            mask_list.append(mask)

        return dict(
            rule_logits=torch.cat(rule_logits_list, dim=0),
            kge_scores=torch.cat(kge_list, dim=0),
            all_h=torch.cat(all_h_list, dim=0),
            all_r=torch.cat(all_r_list, dim=0),
            all_t=torch.cat(all_t_list, dim=0),
            flag=torch.cat(flag_list, dim=0),
            mask=torch.cat(mask_list, dim=0),
            entity_size=test_set.graph.entity_size,
        )

    def _mrr_from_logits(self, scores_dict, alpha, expectation=True):
        """Compute MRR+Hit@K from pre-collected score tensors and a given alpha.

        scores_dict is the dict returned by _collect_kge_scores().
        Combined logit = rule_logits + alpha * kge_scores.
        Returns (mrr, hit1, hit3, hit10, mr).
        """
        rule_logits = scores_dict['rule_logits']
        kge_scores  = scores_dict['kge_scores']
        concat_logits = rule_logits + alpha * kge_scores
        concat_all_t  = scores_dict['all_t']
        concat_flag   = scores_dict['flag']
        concat_mask   = scores_dict['mask']
        entity_size   = scores_dict['entity_size']

        ranks = []
        for k in range(concat_all_t.size(0)):
            t = concat_all_t[k]
            if concat_mask[k, t].item():
                val = concat_logits[k, t]
                L = (concat_logits[k][concat_flag[k]] > val).sum().item() + 1
                H = (concat_logits[k][concat_flag[k]] >= val).sum().item() + 2
            else:
                L = 1
                H = entity_size + 1
            ranks.append((L, H))

        hit1 = hit3 = hit10 = mr = mrr = 0.0
        for (L, H) in ranks:
            if expectation:
                for rank in range(L, H):
                    w = 1.0 / (H - L)
                    if rank <= 1:  hit1  += w
                    if rank <= 3:  hit3  += w
                    if rank <= 10: hit10 += w
                    mr  += rank * w
                    mrr += w / rank
            else:
                rank = H - 1
                if rank <= 1:  hit1  += 1
                if rank <= 3:  hit3  += 1
                if rank <= 10: hit10 += 1
                mr  += rank
                mrr += 1.0 / rank

        n = len(ranks)
        return mrr / n, hit1 / n, hit3 / n, hit10 / n, mr / n

    @torch.no_grad()
    def evaluate_t(self, split, alpha=3.0, expectation=True):
        """KGE-fused evaluation at a fixed alpha. Delegates to _collect_kge_scores
        + _mrr_from_logits so the collection and ranking logic live in one place."""
        logging.info('>>>>> Predictor: Evaluating on {}'.format(split))
        scores_dict = self._collect_kge_scores(split)
        mrr, hit1, hit3, hit10, mr = self._mrr_from_logits(scores_dict, alpha, expectation)

        logging.info('Data : {}'.format(len(scores_dict['all_t'])))
        logging.info('Hit1 : {:.6f}'.format(hit1))
        logging.info('Hit3 : {:.6f}'.format(hit3))
        logging.info('Hit10: {:.6f}'.format(hit10))
        logging.info('MR   : {:.6f}'.format(mr))
        logging.info('MRR  : {:.6f}'.format(mrr))

        return mrr

    @torch.no_grad()
    def evaluate_t_with_sweep(self, split, args, expectation=True):
        """If --alpha_sweep is set, pick the best alpha on the VALID split (one
        forward pass, then cheap ranking over the alpha grid), then re-report on
        `split` at that alpha. Otherwise fall back to evaluate_t at args.alpha.

        The valid-set sweep ensures alpha is never tuned on the test set.
        """
        if not getattr(args, 'alpha_sweep', False):
            return self.evaluate_t(split, float(args.alpha), expectation)

        # Collect valid-set scores once.
        logging.info('[alpha_sweep] Collecting valid-set scores for alpha grid search...')
        valid_scores = self._collect_kge_scores('valid')

        alpha_values = [float(a.strip())
                        for a in str(getattr(args, 'alpha_grid', '0,0.5,1,1.5,2,3,4,6,8')).split(',')]
        best_alpha = float(args.alpha)
        best_valid_mrr = -1.0
        for a in alpha_values:
            mrr, _, _, _, _ = self._mrr_from_logits(valid_scores, a, expectation)
            logging.info('[alpha_sweep] alpha=%.2f -> valid MRR=%.6f' % (a, mrr))
            if mrr > best_valid_mrr:
                best_valid_mrr = mrr
                best_alpha = a

        logging.info('[alpha_sweep] Best alpha=%.2f (valid MRR=%.6f)' % (best_alpha, best_valid_mrr))

        # Report the target split at the best alpha.
        logging.info('>>>>> Predictor: Evaluating on {} (alpha={})'.format(split, best_alpha))
        scores_dict = self._collect_kge_scores(split)
        mrr, hit1, hit3, hit10, mr = self._mrr_from_logits(scores_dict, best_alpha, expectation)

        logging.info('Data : {}'.format(len(scores_dict['all_t'])))
        logging.info('Hit1 : {:.6f}'.format(hit1))
        logging.info('Hit3 : {:.6f}'.format(hit3))
        logging.info('Hit10: {:.6f}'.format(hit10))
        logging.info('MR   : {:.6f}'.format(mr))
        logging.info('MRR  : {:.6f}'.format(mrr))

        return mrr


    def save(self, args, checkpoint):
        """
        Save checkpoint to file.
        Parameters:
            checkpoint (file-like): checkpoint file
        """
        # if comm.get_rank() == 0:
        logging.info("Save checkpoint to %s" % checkpoint)
        checkpoint = os.path.expanduser(checkpoint)
       
        state = {
            "model": self.model.state_dict(),
        }
       
        torch.save(state, checkpoint)

        g_rule_embedding = self.model.mlp_feature.detach().cpu().numpy()
        np.save(
            os.path.join(args.save_path, 'g_rule_embedding'), 
            g_rule_embedding
        )

        # Persist the FM per-rule embedding for offline interaction analysis
        # (also present in grounding.pt as a Parameter; this is a convenience
        # dump mirroring g_rule_embedding).
        if getattr(self.model, 'use_fm', False):
            np.save(
                os.path.join(args.save_path, 'rule_fm_emb'),
                self.model.rule_fm_emb.detach().cpu().numpy()
            )
        
        if getattr(self.model, 'use_nam', False):
            np.save(os.path.join(args.save_path, 'nam_emb'),
                self.model.nam_emb.detach().cpu().numpy())



def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))