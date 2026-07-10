
from utils import *
import torch
from torch import nn

from torch.utils.data import DataLoader
from itertools import islice
from data import Iterator, RuleDataset, KGETrainDataset, BidirectionalOneShotIterator
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

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
        # w_R is warm-started from the raw RulE confidence (gamma_rule - d, Eq. 6)
        # and stored as a frozen buffer -- NO sigmoid -- so the additive score
        # matches the paper's "sum (w/o MLP)" baseline exactly.
        if getattr(args, 'logreg_binary', False):
            # === Learned logistic-regression x binary aggregation ===
            # Per-rule weight = logistic-regression coefficient beta_R (from
            # rule_logreg.pt), fitted by BCE on a leave-one-out train design
            # matrix. Same in-model score form as --precision_binary --
            # binary activation, no per-entity bias -- but with LEARNED weights
            # instead of frozen PCA precision:
            #     score(t) = sum_R beta_R * 1[rule R fired h->t].
            # The fitted intercept b is a constant added to every candidate tail
            # within a query, so it does NOT change the per-query ranking; it is
            # therefore omitted here (use_bias=False) exactly like the precision
            # baseline. beta can be negative (a rule that fires predominantly on
            # non-answers), which is fine for an additive score.
            logreg_file = getattr(args, 'logreg_file', None)
            if logreg_file is None:
                raise ValueError(
                    '--logreg_binary requires --logreg_file pointing at a '
                    'rule_logreg.pt (written by scripts/rule_logreg_train.py).')
            logging.info('[logreg_binary] loading learned weights from %s'
                         % logreg_file)
            lr_data = torch.load(logreg_file, map_location=self.device,
                                 weights_only=False)
            beta = lr_data['beta'].float()
            R = self.model.rule_features.size(0)
            if beta.size(0) != R:
                raise ValueError(
                    'rule_logreg.pt has %d entries but the model has %d rules. '
                    'Ensure both come from the same dataset/rule_file.'
                    % (beta.size(0), R))
            # Align beta to model rule rows via the global rule id stored in
            # rule_features[:, 0]; NaN (should not occur -- unfired rules get
            # beta 0 during the fit) -> 0.0 for safety.
            gids = self.model.rule_features[:, 0].long().to(beta.device)
            w = torch.nan_to_num(beta[gids], nan=0.0)
            intercept = float(lr_data.get('intercept',
                                          torch.tensor(0.0)).item())
            n_pos = int((w > 0).sum().item())
            n_neg = int((w < 0).sum().item())
            self.model.register_buffer('rule_weight_logit',
                                       w.to(self.device).clone())
            self.model.simple_aggregation = True
            self.model.paper_sum = True          # binary activation
            self.model.use_bias = False          # intercept omitted (rank-invariant)
            self.model.bias.requires_grad_(False)
            logging.info('[logreg_binary] ENABLED: binary activation, no bias, '
                         'learned per-rule beta '
                         '(pos:%d neg:%d /%d, intercept=%.4f omitted from ranking).'
                         % (n_pos, n_neg, R, intercept))
        elif getattr(args, 'precision_binary', False):
            # === Precision x binary aggregation ===
            # Per-rule weight = empirical train PCA precision (from
            # rule_precision.pt) instead of the frozen RulE confidence. Binary
            # activation (paper_sum) and no bias, so the score is
            # sum_R precision_R * 1[rule R fired on t]. Precision is in [0, 1],
            # so no clamping is needed. This mirrors the offline scorer
            # score_counts_offline.py --weight_source precision.
            precision_file = getattr(args, 'precision_file', None)
            if precision_file is None:
                raise ValueError(
                    '--precision_binary requires --precision_file pointing at a '
                    'rule_precision.pt (written by scripts/rule_precision_train.py).')
            logging.info('[precision_binary] loading precision weights from %s'
                         % precision_file)
            prec_data = torch.load(precision_file, map_location=self.device,
                                   weights_only=False)
            precision = prec_data['precision'].float()
            R = self.model.rule_features.size(0)
            if precision.size(0) != R:
                raise ValueError(
                    'rule_precision.pt has %d entries but the model has %d rules. '
                    'Ensure both come from the same dataset/rule_file.'
                    % (precision.size(0), R))
            # Align precision to model rule rows via the global rule id stored in
            # rule_features[:, 0]; NaN (rules that never fired from a known
            # subject) -> 0.0 so they contribute nothing, matching the offline
            # scorer.
            gids = self.model.rule_features[:, 0].long().to(precision.device)
            w = torch.nan_to_num(precision[gids], nan=0.0)
            n_nan = int(torch.isnan(precision[gids]).sum().item())
            n_pos = int((w > 0).sum().item())
            self.model.register_buffer('rule_weight_logit',
                                       w.to(self.device).clone())
            self.model.simple_aggregation = True
            self.model.paper_sum = True          # binary activation
            self.model.use_bias = False          # purely rule-based
            self.model.bias.requires_grad_(False)
            logging.info('[precision_binary] ENABLED: binary activation, no bias, '
                         'per-rule precision weights '
                         '(rules>0: %d/%d, nan->0: %d).' % (n_pos, R, n_nan))
        elif getattr(args, 'simple_aggregation', False):
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
            if getattr(args, 'clamp_negative_confidence', False):
                n_neg = int((init_logits < 0).sum().item())
                init_logits = init_logits.clamp(min=0.0)
                logging.info('[clamp] zeroed %d/%d negative-confidence rules '
                             '(w_R = max(0, w_R)).' % (n_neg, init_logits.numel()))
            self.model.register_buffer('rule_weight_logit', init_logits.clone())
            self.model.simple_aggregation = True
            self.model.paper_sum = getattr(args, 'paper_sum', False)
            # Bias is dropped under --paper_sum (faithful paper sum) or --no_bias
            # (counts, purely rule-based). In either case freeze it so it picks up
            # no gradient signal and stays at its init value.
            self.model.use_bias = not (self.model.paper_sum
                                       or getattr(args, 'no_bias', False))
            if not self.model.use_bias:
                self.model.bias.requires_grad_(False)
            if self.model.paper_sum:
                logging.info('[paper_sum] faithful paper sum ENABLED: binary activation, '
                             'no bias, raw frozen RulE confidence.')
            elif not self.model.use_bias:
                logging.info('[additive] simple aggregation, NO bias (raw counts, '
                             'purely rule-based; num_rules=%d).' % init_logits.numel())
            else:
                logging.info('[additive] simple aggregation with FROZEN per-rule weight '
                             '(RAW RulE confidence, no sigmoid; num_rules=%d).' % init_logits.numel())

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=float(args.g_lr),
            weight_decay=float(args.weight_decay))

        self.train_set.make_batches()
        
        train_dataloader = DataLoader(self.train_set, 1, num_workers=self.num_worker)

        # rules_weight_emb (the per-dim MLP rule message) is only consumed by the
        # original MLP forward path. The additive/paper_sum path returns before
        # touching it, so skip this precompute there.
        if not getattr(self.model, 'simple_aggregation', False):
            self.model.eval_compute_rule_weight(self.device)

        logging.info('>>>>> RulE: Grounding-Training')

        best_valid_mrr = 0.0 
        test_mrr = 0.0

        for k in range(args.num_iters):

            logging.info('-------------------------')
            logging.info('| Iteration: {}/{}'.format(k + 1, args.num_iters))
            logging.info('-------------------------')

            self.train_step(optimizer, train_dataloader, args.batch_per_epoch, args.smoothing, args.print_every, args)
            valid_mrr_iter = self.evaluate('valid', args.alpha, expectation=True)

            if valid_mrr_iter > best_valid_mrr:
                best_valid_mrr = valid_mrr_iter
                test_mrr = valid_mrr_iter
                self.save(args, os.path.join(args.save_path, 'grounding.pt'))

        logging.info('-------------------------')
        logging.info('| Best Valid MRR (model selection): {:.6f}'.format(test_mrr))
        logging.info('-------------------------')

        # Reload best-valid checkpoint so final eval reflects the selected model.
        # grounding.pt is only written when valid MRR improves over the initial
        # 0.0, so it may be missing if no iteration ever improved (e.g. a fully
        # frozen paper_sum run where no rule fires). Fall back to the current
        # in-memory model in that case.
        best_ckpt = os.path.join(self.args.save_path, 'grounding.pt')
        if os.path.exists(best_ckpt):
            checkpoint = torch.load(best_ckpt)
            self.model.load_state_dict(checkpoint['model'])
        else:
            logging.info('[eval] No best-valid checkpoint at %s (valid MRR never '
                         'improved); evaluating current model state.' % best_ckpt)

        test_mrr_iter = self.evaluate('valid', args.alpha, expectation=True)
        test_mrr_iter = self.evaluate('test', args.alpha, expectation=True)
        test_mrr_iter = self.evaluate_t('test_kge', args.alpha, expectation=True)


       
       

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

                # Under --paper_sum every term in the grounding score is a frozen
                # buffer or a no_grad grounding constant, so `loss` has no
                # grad_fn and backward() would raise. Skip the optimisation step
                # in that case. NOTE: loss.requires_grad tests reachability, which
                # is the correct check (requires_grad on a parameter alone does
                # not guarantee the parameter appears in the loss graph).
                if loss.requires_grad:
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

        # Per-query rank dump (h, r, t, L, H) for offline rank-win comparison.
        if getattr(self.args, 'dump_ranks', False):
            self._dump_ranks(split, query2LH)

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


    def _dump_ranks(self, split, query2LH):
        """Write per-query filtered ranks to ranks_<split>.csv in save_path.

        Columns: h, r, t, L, H. Rank = (L + H - 1) / 2 is the standard
        mid-point for tie bands; downstream comparison recomputes it.
        """
        import csv
        out_path = os.path.join(self.args.save_path, 'ranks_%s.csv' % split)
        with open(out_path, 'w', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow(['h', 'r', 't', 'L', 'H'])
            for (h, r, t), (L, H) in query2LH.items():
                writer.writerow([h, r, t, L, H])
        logging.info('[dump_ranks] wrote %s (%d queries)'
                     % (out_path, len(query2LH)))


    @torch.no_grad()
    def evaluate_t(self, split, alpha=3.0, expectation=True):

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

            logits, mask = model(all_h, all_r, None)

            kge_score = model.compute_g_KGE(all_h, all_r)
            logits = logits + alpha * kge_score

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


    def save(self, args, checkpoint):
        """
        Save checkpoint to file.
        Parameters:
            checkpoint (file-like): checkpoint file
        """
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



def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))