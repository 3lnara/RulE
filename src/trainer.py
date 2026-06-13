
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

            KGE_score = model.compute_g_KGE(all_h,all_r)
            
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

        self.model.load_state_dict(state["model"], strict=False)

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

        # Diagnostic: optionally freeze the per-rule confidence head. If MRR
        # stops degrading, conf_proj.bias drift was the cause; otherwise the
        # GAT itself (W_src/W_dst/W_rel/attn_vec) is.
        if getattr(args, 'freeze_conf_proj', False):
            self.model.grounding_gat.conf_proj.weight.requires_grad = False
            if self.model.grounding_gat.conf_proj.bias is not None:
                self.model.grounding_gat.conf_proj.bias.requires_grad = False
            logging.info('[diagnostic] grounding_gat.conf_proj is FROZEN '
                         '(weight + bias requires_grad=False).')

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=float(args.g_lr), 
            weight_decay=float(args.weight_decay))


        self.train_set.make_batches()
        
        train_dataloader = DataLoader(self.train_set, 1, num_workers=self.num_worker)
        
        self.model.eval_compute_rule_weight(self.device)
        
        if getattr(args, 'use_rule_confidence_variant_a', False):
            rule_features = self.model.rule_features.to(self.device)
            rule_masks = self.model.rule_masks.to(self.device)
            batch = 128
            split_num = max(1, rule_features.size(0) // batch)
            scalars = []
            with torch.no_grad():
                for rules, masks in zip(
                    torch.split(rule_features, split_num, 0),
                    torch.split(rule_masks, split_num, 0),
                ):
                    scores, _ = self.model.add_ruleE(rules.unsqueeze(1), masks)
                    scalars.append(scores.squeeze(1))
            scalar_confidences = torch.sigmoid(torch.cat(scalars, dim=0))
            self.model.grounding_gat.set_frozen_confidences(scalar_confidences)
        
        logging.info('>>>>> RulE: Grounding-Training')

        if getattr(args, 'checkpoint_grounding', False):
            logging.info('[memory] checkpoint_grounding=True: GAT edge-attention is recomputed '
                         'in backward (exact gradients, lower memory).')
            if float(getattr(args, 'attn_entropy_weight', 0.0)) > 0.0:
                logging.warning('attn_entropy_weight>0 is IGNORED while checkpoint_grounding=True '
                                '(the entropy term needs the attention graph that checkpointing '
                                'frees). Disable checkpointing to use attention-entropy.')
        

        best_valid_mrr = 0.0 
        test_mrr = 0.0

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

            self.train_step( optimizer, train_dataloader, args.batch_per_epoch, args.smoothing, args.print_every, args, iter_idx=k)
            valid_mrr_iter = self.evaluate('valid', args.alpha, expectation=True)
            # test_mrr_iter = self.evaluate('test', args.alpha, expectation=True)
            # test_mrr_iter = self.evaluate_t('test_kge', args.alpha, expectation=True)
            

            if valid_mrr_iter > best_valid_mrr:
                best_valid_mrr = valid_mrr_iter
                test_mrr = valid_mrr_iter  # Fixed: was referencing undefined test_mrr_iter
                self.save(args, os.path.join(args.save_path, 'grounding.pt'))
        

        logging.info('-------------------------')
        logging.info('| Final Test MRR: {:.6f}'.format(test_mrr))
        logging.info('-------------------------')

        checkpoint = torch.load(os.path.join(self.args.save_path, 'grounding.pt'))
        self.model.load_state_dict(checkpoint['model'], strict=False)
        
        test_mrr_iter = self.evaluate('valid', args.alpha, expectation=True)
        test_mrr_iter = self.evaluate('test', args.alpha, expectation=True)
        test_mrr_iter = self.evaluate_t('test_kge', args.alpha, expectation=True)
       

    def train_step(self, optimizer, train_dataloader, batch_per_epoch, smoothing, print_every, args, iter_idx=0):

        batch_per_epoch = batch_per_epoch or len(train_dataloader)
        model = self.model

        model.train()

        total_loss = 0.0
        total_size = 0.0

        # Diagnostic flags (read once at top so the inner loop is cheap).
        attn_entropy_weight = float(getattr(args, 'attn_entropy_weight', 0.0))
        eval_every = int(getattr(args, 'eval_every_batches', 0))
        do_per_batch_eval = (iter_idx == 0 and eval_every > 0)
        if do_per_batch_eval:
            logging.info(f'[diagnostic] iter 1: validating every {eval_every} batches')
        if attn_entropy_weight > 0.0:
            logging.info(f'[diagnostic] attention-entropy regularisation \u03bb={attn_entropy_weight}')

        optimizer.zero_grad(set_to_none=True)

        for batch_id, batch in enumerate(islice(train_dataloader, batch_per_epoch)):
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
            if batch_id == 0:
                for name, module in self.model.grounding_gat.named_modules():
                    if hasattr(module, 'last_attn'):  # only if you cache it
                        attn = module.last_attn
                        logging.info(f"attn std={attn.std():.4f}, min={attn.min():.4f}, max={attn.max():.4f}")
            if mask.sum().item() != 0:
                rule_logits =  torch.nn.functional.log_softmax(grounding_rule_score, dim=1)

                ce_loss = -(rule_logits[mask] * target[mask]).sum() / torch.clamp(target[mask].sum(), min=1)

                # Optional: encourage uniform attention to counter collapse.
                # Σ a*log(a) is most negative when uniform, ~0 when peaked, so
                # adding a positive multiple of it to the loss rewards spread.
                entropy_term = self.model.grounding_gat.attn_entropy_penalty
                if attn_entropy_weight > 0.0 and entropy_term is not None:
                    loss = ce_loss + attn_entropy_weight * entropy_term
                else:
                    loss = ce_loss

                loss.backward()

                if batch_id == 0:
                    for name, param in self.model.grounding_gat.named_parameters():
                        if param.grad is None:
                            logging.info(f"NO GRAD: {name}")
                        else:
                            logging.info(f"{name}: grad_norm={param.grad.norm().item():.6f}")
                    if attn_entropy_weight > 0.0 and entropy_term is not None:
                        logging.info(f"attn_entropy_penalty (\u03a3 a*log a) = {entropy_term.item():.4f} "
                                     f"(more negative \u21d2 more uniform)")

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()

                total_loss += loss.item()
                total_size += mask.sum().item()
            
            if (batch_id + 1) % print_every == 0:
                
                
                logging.info('loss:    {} {} {:.6f} {:.1f}'.format(batch_id + 1, len(train_dataloader), loss, total_size / print_every))
                
                total_loss = 0.0
                total_size = 0.0
                self.save(args, os.path.join(args.save_path, 'grounding.pt'))

            # Per-batch validation in iteration 1 only. Tracks whether MRR is
            # already at its peak after batch 0 (i.e. all subsequent training
            # is noise) or whether the GAT/conf_proj actually improves it.
            if do_per_batch_eval and (batch_id + 1) % eval_every == 0:
                logging.info(f'[per-batch-eval] iter=1 batch={batch_id + 1} \u2014 evaluating on valid')
                mrr = self.evaluate('valid', args.alpha, expectation=True)
                logging.info(f'[per-batch-eval] iter=1 batch={batch_id + 1} valid_mrr={mrr:.6f}')
                model.train()
        

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

            # logits, mask = model.forward_weight(all_h, all_r, None)
            logits, mask = model(all_h, all_r, None)

            # Add KGE score with fixed alpha (original approach)
            if alpha is not None and alpha > 0:
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
        # if comm.get_rank() == 0:
        logging.info("Save checkpoint to %s" % checkpoint)
        checkpoint = os.path.expanduser(checkpoint)
       
        state = {
            "model": self.model.state_dict(),
        }
       
        torch.save(state, checkpoint)

        if hasattr(self.model, 'mlp_feature'):
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