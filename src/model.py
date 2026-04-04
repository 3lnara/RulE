
import torch
import torch.nn as nn
import logging, math
from layers import MLP, FuncToNodeSum, GroundingGAT

from torch.nn.utils.rnn import pad_sequence

class RulE(torch.nn.Module):
    def __init__(self, graph, p_norm, mlp_rule_dim, gamma_fact, gamma_rule, hidden_dim, device, dataset):
        super(RulE, self).__init__()
        self.graph = graph
        self.device = device
        self.num_entities = graph.entity_size
        self.num_relations = graph.relation_size 
        self.padding_index = graph.relation_size 

        self.hidden_dim = hidden_dim
        # self.entity_dim = hidden_dim * 2 
        # self.relation_dim = hidden_dim

        # self.rule_dim = rule_dim
        # self.rule_dim = self.relation_dim

        self.p = p_norm

        self.mlp_rule_dim = mlp_rule_dim

        self.grounding_gat = GroundingGAT(
            entity_dim=self.hidden_dim * 2,
            relation_dim=self.hidden_dim,
        )

        self.bias = torch.nn.parameter.Parameter(torch.zeros(self.num_entities))
        
        # === Learnable per-relation beta for KGE/Rule balancing ===
        # beta * rule_score + (1-beta) * kge_score
        # Each relation learns its optimal balance during training/fine-tuning
        # Can be trained during grounding phase by freezing all other parameters
        self.beta = nn.Parameter(torch.zeros(self.num_relations * 2))
        nn.init.constant_(self.beta, 0.0)  # sigmoid(0) = 0.5, starts at equal weighting
        
        # === Query-adaptive beta: conditions on grounding density ===
        # beta(h,r) = sigmoid(beta_rel[r] + beta_density * density(h,r))
        # Sparse queries (few groundings) -> lower beta -> trust KGE more
        # Dense queries (many groundings) -> higher beta -> trust rules more
        self.beta_density = nn.Parameter(torch.zeros(1))
        
        self.epsilon = 2.0

        
        self.gamma_fact = nn.Parameter(
            torch.Tensor([gamma_fact]), 
            requires_grad=False
        )

        self.gamma_rule = nn.Parameter(
            torch.Tensor([gamma_rule]), 
            requires_grad=False
        )
        
        self.embedding_range_fact = nn.Parameter(
            torch.Tensor([(self.gamma_fact.item() + self.epsilon) / hidden_dim]), 
            requires_grad=False
        )
        
        self.embedding_range_rule = nn.Parameter(
            torch.Tensor([(self.gamma_rule.item() + self.epsilon) / hidden_dim]), 
            requires_grad=False
        )

        self.entity_embedding = torch.nn.Embedding(self.num_entities, self.hidden_dim * 2)
        # nn.init.ones_(
        #     tensor=self.entity_embedding.weight
        # )
        nn.init.uniform_(
            tensor=self.entity_embedding.weight, 
            a=-self.embedding_range_fact.item(), 
            b=self.embedding_range_fact.item()
        )
        
        self.relation_embedding = torch.nn.Embedding(self.num_relations + 1, self.hidden_dim, padding_idx=self.padding_index)
        # nn.init.ones_(
        #     tensor=self.relation_embedding.weight
        # )
        nn.init.uniform_(
            tensor=self.relation_embedding.weight, 
            a=-self.embedding_range_fact.item(), 
            b=self.embedding_range_fact.item()
        )

        # # Initialize to 1
        # nn.init.zeros_(
        #     tensor=self.relation_embedding.weight[self.padding_index]
        # )
        
        # RNN parameters
        # self.rnn_hidden_dim = rnn_hidden_dim
        # self.num_layers = num_layers
        # self.rnn = torch.nn.LSTM(self.relation_dim + self.rule_dim, self.rnn_hidden_dim, self.num_layers, batch_first=True)
        # self.linear = torch.nn.Linear(self.rnn_hidden_dim, self.relation_dim)
        
        self.pi = 3.14159262358979323846

    # def add_param(self):

    #     # self.mlp_rule_dim = 16
    #     self.mlp_feature = nn.Parameter(torch.zeros(self.num_rules, self.mlp_rule_dim))
    #     # nn.init.kaiming_uniform_(self.mlp_feature, a=math.sqrt(5), mode="fan_in")
        
    #     # self.beta = nn.Parameter(torch.zeros((self.num_relations * 2)))
    #     # torch.nn.init.uniform_(self.beta, a=0, b=1)
        
    #     self.rule_to_entity = FuncToNodeSum(self.mlp_rule_dim)

    #     # self.relation_emb = torch.nn.Embedding(self.num_relations, self.mlp_rule_dim)
    #     self.score_model = MLP(self.mlp_rule_dim, [128, 1]) # 128 for FB15k
        
    #     # if self.device.type == "cuda":
    #     #     self.score_model = self.score_model.cuda(self.device)
    #     #     self.rule_to_entity = self.rule_to_entity.cuda(self.device)

    def set_rules(self, input):
        # input: [rule_id, rule_head, rule_body]

        logging.info('read {} rules from list.'.format(len(input)))
        self.num_rules = len(input)

        # rule_body's length
        self.max_length = max([len(rule[2:]) for rule in input])

        # self.rule_dim = self.hidden_dim * self.max_length
        self.rule_dim = self.hidden_dim 
        
        self.relation2rules = [[] for r in range(self.num_relations*2)]
        for rule in input:
            relation = rule[1]
            self.relation2rules[relation].append([rule[0], (rule[1], rule[2:])])
        

        self.rule_features = []
        rule_masks = list()
        for rule in input:
            rule_ = rule + [self.padding_index for i in range(self.max_length - len(rule[2:]))]
            self.rule_features.append(rule_)
            rule_mask = torch.ones_like(torch.tensor(rule))[2:].bool()

            # self.rule_mask = torch.zeros_like(torch.tensor(rule_))[2:].bool()
            # self.rule_mask[(len(rule[2:]))-1] = True
            rule_masks.append(rule_mask)

        # self.rule_masks = torch.stack(self.rule_masks)
        self.rule_masks = pad_sequence([_ for _ in rule_masks], batch_first=True,padding_value=False)
        self.rule_features = torch.tensor(self.rule_features, dtype=torch.long)


        self.rule_emb = torch.nn.Embedding(self.num_rules, self.rule_dim)
        nn.init.kaiming_uniform_(self.rule_emb.weight, a=math.sqrt(5), mode="fan_in")
        # nn.init.uniform_(
        #     tensor=self.rule_emb.weight, 
        #     a=-self.embedding_range_rule.item(), 
        #     b=self.embedding_range_rule.item()
        # )
        
       
    def compute_ruleE(self, sample, mode='single'):

        if mode == 'single':
            rule, mask = sample
           
            score, rule_emb = self.add_ruleE(rule.unsqueeze(1), mask)

        elif mode == 'batch':
            pos_part, mask,  neg_idx, neg_part = sample
            batch_size, negative_sample_size = neg_idx.size(0), neg_idx.size(1)
            
            pos_part = pos_part.unsqueeze(dim=1).repeat(1,negative_sample_size,1)
            
            neg_idx = neg_idx.unsqueeze(dim=2) + 1
            neg_part = neg_part.unsqueeze(dim=2)
            rule_sample = pos_part.scatter(2, neg_idx, neg_part)
           
            score, rule_emb = self.add_ruleE(rule_sample, mask)
            
        return score



    def compute_KGE(self, sample, mode='single'):
        
        if mode == 'single':

            head = self.entity_embedding(sample[:,0]).unsqueeze(1)

            relation = self.relation_embedding(sample[:,1]).unsqueeze(1)
            
            tail = self.entity_embedding(sample[:,2]).unsqueeze(1)

        elif mode == 'head-batch':
            tail_part, head_part = sample

            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            
            head = self.entity_embedding(head_part.view(-1)).view(batch_size, negative_sample_size, -1)
            relation = self.relation_embedding(tail_part[:,1]).unsqueeze(1)
            tail = self.entity_embedding(tail_part[:,2]).unsqueeze(1)

        elif mode == 'tail-batch':

            head_part, tail_part = sample

            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            
            head = self.entity_embedding(head_part[:,0]).unsqueeze(1)
            relation = self.relation_embedding(head_part[:,1]).unsqueeze(1)
            tail = self.entity_embedding(tail_part.view(-1)).view(batch_size, negative_sample_size, -1)

        else:
            raise ValueError('mode %s not supported' % mode)

        return self.RotatE(head,relation,tail,mode), (head, tail)

    

    def compute_g_KGE(self,all_h,all_r):

        all_t = torch.arange(0,self.num_entities,device=all_h.device).unsqueeze(0).repeat(all_h.size(0),1)
       
        relations_flag = torch.pow(-1, all_r // self.num_relations).unsqueeze(-1)
        all_r = all_r % (self.num_relations)

        head = self.entity_embedding(all_h).unsqueeze(1)

        relation = (self.relation_embedding(all_r) * relations_flag).unsqueeze(1)

        tail = self.entity_embedding(all_t.view(-1)).view(all_h.size(0), self.num_entities, -1)
        

        return self.RotatE(head,relation,tail)


    def RotatE(self, head, relation, tail, mode='tail-batch'):
       
        
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        #Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation/(self.embedding_range_fact.item()/self.pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail
      

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)
        
        score = self.gamma_fact.item() - score.sum(dim=2)
        
        return score
    


    def add_ruleE(self, rules, mask):
        inputs = rules[:,:,2:]
        # cal_mask = (~mask).unsqueeze(1).unsqueeze(-1)
        rule_len = mask.sum(-1).unsqueeze(1).unsqueeze(-1)
        cal_mask = mask.unsqueeze(1).unsqueeze(-1)
        relations_flag = torch.pow(-1,inputs // (self.num_relations)).unsqueeze(-1)
        inputs_com = inputs % self.num_relations

        inputs_com = torch.where(inputs==self.num_relations * 2, self.padding_index, inputs_com)
        
        embedding = self.relation_embedding(inputs_com) * relations_flag
        
        rule_embedding = self.rule_emb(rules[:,:,0])
        
        
        # rule_head
        embedding_r = self.relation_embedding(rules[:,:,1]%self.num_relations)
        relations_flag = torch.pow(-1,rules[:,:,1] // (self.num_relations)).unsqueeze(-1)
        embedding_r *= relations_flag

        rule_body = embedding * cal_mask
        
        
        
        outputs = rule_body.sum(-2) + rule_embedding


        # dist = self.gamma_rule.item() - torch.norm((outputs - embedding_r), dim=-1)
        dist = self.gamma_rule.item() - torch.norm((outputs - embedding_r), p=self.p, dim=-1)

        
        return dist, rule_embedding
    


    def add_ruleE_g(self, rules, mask):
        inputs = rules[:,:,2:]
        # cal_mask = (~mask).unsqueeze(1).unsqueeze(-1)
        rule_len = mask.sum(-1).unsqueeze(1).unsqueeze(-1)
        cal_mask = mask.unsqueeze(1).unsqueeze(-1)
        relations_flag = torch.pow(-1,inputs // (self.num_relations)).unsqueeze(-1)
        inputs_com = inputs % self.num_relations

        inputs_com = torch.where(inputs==self.num_relations * 2, self.padding_index, inputs_com)
        
        embedding = self.relation_embedding(inputs_com) * relations_flag
        
        rule_embedding = self.rule_emb(rules[:,:,0])
        
        
        # rule_head
        embedding_r = self.relation_embedding(rules[:,:,1]%self.num_relations)
        relations_flag = torch.pow(-1,rules[:,:,1] // (self.num_relations)).unsqueeze(-1)
        embedding_r *= relations_flag

        rule_body = embedding * cal_mask
        
        
        # outputs = rule_body.sum(-2) + rule_embedding
        outputs = rule_body.sum(-2) + rule_embedding

        dist = self.gamma_rule.item()/self.hidden_dim - torch.pow((outputs - embedding_r), self.p)
        # dist = self.gamma_rule.item() - torch.norm((outputs - embedding_r), p = self.p, dim=-1)

        return dist
    

    def forward(self, all_h, all_r, edges_to_remove, return_attention=False):
        query_r = all_r[0].item()
        assert (all_r != query_r).sum() == 0
        device = all_r.device

        if device.type == "cuda":
            self.rule_features = self.rule_features.cuda(device)

        entity_emb = self.entity_embedding.weight
        relation_emb = self.relation_embedding.weight

        score = torch.zeros(all_h.size(0), self.graph.entity_size, device=device)
        mask = torch.zeros(all_h.size(0), self.graph.entity_size, device=device)

        attention_info = [] if return_attention else None

        for index, (r_head, r_body) in self.relation2rules[query_r]:
            assert r_head == query_r

            attn_count, hop_attentions = self.graph.grounding_with_attention(
                all_h, r_head, r_body,
                self.grounding_gat.compute_edge_attention,
                entity_emb, relation_emb,
                edges_to_remove,
            )

            rule_emb_vec = self.rules_weight_emb[index]
            confidence = self.grounding_gat.compute_confidence(rule_emb_vec)

            rule_score = attn_count * confidence
            score = score + rule_score
            mask = mask + (attn_count > 0).float()

            if return_attention:
                attention_info.append({
                    'rule_index': index,
                    'rule_head': r_head,
                    'rule_body': r_body,
                    'confidence': confidence.item(),
                    'attn_grounding': attn_count.detach(),
                    'hop_attentions': hop_attentions,
                })

        if mask.sum().item() == 0:
            if return_attention:
                return mask + self.bias.unsqueeze(0), (1 - mask).bool(), []
            return mask + self.bias.unsqueeze(0), (1 - mask).bool()

        score = score + self.bias.unsqueeze(0)
        mask = torch.ones_like(mask).bool()

        if return_attention:
            return score, mask, attention_info
        return score, mask




    def eval_compute_rule_weight(self,device):
        '''
        During grounding process, we one time compute the rule score on rule embedding and relation embedding
        '''
        batch = 128
        self.rule_masks = self.rule_masks.to(device)
        self.rule_features = self.rule_features.to(device)
        split_num = self.rule_features.size(0) // batch 
        rule_batches = torch.split(self.rule_features, split_num, 0)
        rule_mask_batches = torch.split(self.rule_masks, split_num, 0)
        rules_weight_emb = list()

        for rules, rules_mask in zip(rule_batches, rule_mask_batches):

            rule_weight_emb = self.add_ruleE_g(rules.unsqueeze(1),rules_mask).squeeze(1)
            rules_weight_emb.append(rule_weight_emb)

        self.rules_weight_emb = torch.cat(rules_weight_emb)

    @torch.no_grad()
    def explain_prediction(self, h, r, t, entity_dict=None, relation_dict=None, top_k=5):
        """
        Explain why the model predicts (h, r, t) by showing per-rule and
        per-path contributions through the grounding graph.

        Args:
            h, r, t: int indices for head, relation, tail
            entity_dict:   optional idx -> name mapping
            relation_dict: optional idx -> name mapping
            top_k: max number of rules to show

        Returns:
            explanation: list of dicts with rule-level and path-level details
        """
        device = next(self.parameters()).device
        all_h = torch.tensor([h], device=device)
        all_r = torch.tensor([r], device=device)

        score, mask, attention_info = self.forward(all_h, all_r, None, return_attention=True)

        if not attention_info:
            return []

        def fmt_rel(idx):
            num_r = self.num_relations
            if relation_dict and idx < len(relation_dict):
                return relation_dict[idx]
            if idx >= num_r:
                base = idx - num_r
                name = relation_dict[base] if (relation_dict and base < len(relation_dict)) else f"r{base}"
                return f"inv({name})"
            return f"r{idx}"

        def fmt_ent(idx):
            if entity_dict and idx < len(entity_dict):
                return entity_dict[idx]
            return f"e{idx}"

        results = []
        for info in attention_info:
            contribution = info['attn_grounding'][0, t].item() * info['confidence']
            if contribution == 0:
                continue

            rule_entry = {
                'rule_index': info['rule_index'],
                'rule_head': fmt_rel(info['rule_head']),
                'rule_body': [fmt_rel(rb) for rb in info['rule_body']],
                'confidence': info['confidence'],
                'contribution': contribution,
                'paths': [],
            }

            # Trace paths through intermediate entities (works for any hop count)
            hops = info['hop_attentions']
            body = info['rule_body']

            def trace_paths(current_nodes, hop_idx, path_entities, path_attns, max_paths=10):
                if hop_idx == len(hops):
                    if t in [n.item() if isinstance(n, torch.Tensor) else n for n in current_nodes]:
                        path_score = 1.0
                        for a in path_attns:
                            path_score *= a
                        rule_entry['paths'].append({
                            'entities': [fmt_ent(e.item() if isinstance(e, torch.Tensor) else e) for e in path_entities] + [fmt_ent(t)],
                            'relations': [fmt_rel(body[i]) for i in range(len(path_attns))],
                            'edge_attentions': list(path_attns),
                            'path_score': path_score,
                        })
                    return

                if len(rule_entry['paths']) >= max_paths:
                    return

                edge_attn, node_in, node_out = hops[hop_idx]
                for src in current_nodes:
                    src_val = src.item() if isinstance(src, torch.Tensor) else src
                    edge_mask = (node_in == src_val)
                    if not edge_mask.any():
                        continue
                    targets = node_out[edge_mask]
                    attns = edge_attn[edge_mask]
                    sorted_idx = attns.argsort(descending=True)[:5]
                    for idx in sorted_idx:
                        dst = targets[idx]
                        a = attns[idx].item()
                        trace_paths(
                            [dst], hop_idx + 1,
                            path_entities + [src_val],
                            path_attns + [a],
                            max_paths,
                        )

            trace_paths([h], 0, [], [], max_paths=5)

            rule_entry['paths'].sort(key=lambda p: p['path_score'], reverse=True)
            results.append(rule_entry)

        results.sort(key=lambda r: r['contribution'], reverse=True)
        return results[:top_k]

    def format_explanation(self, explanation):
        """Format explain_prediction output as a readable string."""
        if not explanation:
            return "No rules were applicable for this prediction."
        lines = []
        for rule in explanation:
            body_str = " /\\ ".join(rule['rule_body'])
            lines.append(
                f"Rule (body: [{body_str}] -> {rule['rule_head']}), "
                f"confidence: {rule['confidence']:.4f}, "
                f"contribution: {rule['contribution']:.4f}"
            )
            for p in rule['paths'][:5]:
                path_str = ""
                for i, (ent, rel, attn) in enumerate(
                    zip(p['entities'][:-1], p['relations'], p['edge_attentions'])
                ):
                    path_str += f"{ent} --({rel}, a={attn:.3f})--> "
                path_str += p['entities'][-1]
                lines.append(f"  Path: {path_str}  score={p['path_score']:.4f}")
            lines.append("")
        return "\n".join(lines)

    def compute_adaptive_beta(self, rule_score, all_r):
        """
        Compute query-adaptive beta that depends on both the relation type
        and how dense the grounding is for this specific query.

        Args:
            rule_score: [batch, num_entities] - rule grounding scores (after forward)
            all_r: [batch] - relation indices

        Returns:
            beta: [batch, 1] - adaptive beta in (0, 1)
            density: [batch] - grounding density per query (for interpretability)
        """
        min_score = rule_score.min(dim=-1, keepdim=True)[0]
        grounded = ((rule_score - min_score) > 1e-6).float()
        density = grounded.mean(dim=-1)                           # [batch]

        logit = self.beta[all_r] + self.beta_density * density    # [batch]
        beta = torch.sigmoid(logit).unsqueeze(-1)                 # [batch, 1]
        return beta, density

    def combine_scores(self, rule_score, kge_score, all_r, method='fixed_alpha', alpha=5.0):
        """
        Combine rule grounding score and KGE score using different methods.
        
        Args:
            rule_score: [batch, num_entities] - score from rule grounding
            kge_score: [batch, num_entities] - score from KGE (RotatE)
            all_r: [batch] - relation indices
            method: str - 'fixed_alpha' or 'learnable_beta'
            alpha: float - fixed alpha value (only used if method='fixed_alpha')
            
        Returns:
            combined_score: [batch, num_entities]
            weight_info: dict with balancing weights for interpretability
        """
        if method == 'fixed_alpha':
            # Original approach: rule_score + alpha * kge_score
            combined = rule_score + alpha * kge_score
            weight_info = {
                'method': 'fixed_alpha',
                'alpha': alpha,
                'rule_weight': 1.0,
                'kge_weight': alpha
            }
            
        elif method == 'learnable_beta':
            # Per-relation learnable: beta * rule_score + (1-beta) * kge_score
            # Use sigmoid to constrain beta to [0, 1]
            beta = torch.sigmoid(self.beta[all_r])  # [batch]
            beta = beta.unsqueeze(-1)  # [batch, 1] for broadcasting
            
            # Normalize scores to similar scale before combining
            rule_norm = rule_score / (rule_score.abs().max(dim=-1, keepdim=True)[0] + 1e-8)
            kge_norm = kge_score / (kge_score.abs().max(dim=-1, keepdim=True)[0] + 1e-8)
            
            combined = beta * rule_norm + (1 - beta) * kge_norm
            
            weight_info = {
                'method': 'learnable_beta',
                'beta_per_relation': beta.squeeze(-1).detach(),  # [batch]
                'rule_weight': beta.squeeze(-1).mean().item(),
                'kge_weight': (1 - beta.squeeze(-1)).mean().item()
            }
            
        elif method == 'learnable_beta_unnorm':
            # Learnable beta without normalization (more direct comparison)
            beta = torch.sigmoid(self.beta[all_r]).unsqueeze(-1)
            combined = beta * rule_score + (1 - beta) * kge_score
            
            weight_info = {
                'method': 'learnable_beta_unnorm',
                'beta_per_relation': beta.squeeze(-1).detach(),
                'rule_weight': beta.squeeze(-1).mean().item(),
                'kge_weight': (1 - beta.squeeze(-1)).mean().item()
            }

        elif method == 'adaptive_beta':
            beta, density = self.compute_adaptive_beta(rule_score, all_r)
            combined = beta * rule_score + (1 - beta) * kge_score

            weight_info = {
                'method': 'adaptive_beta',
                'beta_per_query': beta.squeeze(-1).detach(),
                'density_per_query': density.detach(),
                'rule_weight': beta.squeeze(-1).mean().item(),
                'kge_weight': (1 - beta.squeeze(-1)).mean().item(),
                'beta_density_weight': self.beta_density.item(),
            }
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return combined, weight_info
    
    def get_beta_summary(self):
        """Get summary of learned beta values per relation."""
        with torch.no_grad():
            beta_values = torch.sigmoid(self.beta)
            return {
                'beta_raw': self.beta.detach().cpu().numpy(),
                'beta_sigmoid': beta_values.cpu().numpy(),
                'mean_rule_weight': beta_values.mean().item(),
                'mean_kge_weight': (1 - beta_values).mean().item(),
                'min_beta': beta_values.min().item(),
                'max_beta': beta_values.max().item(),
                'beta_density_weight': self.beta_density.item(),
            }
