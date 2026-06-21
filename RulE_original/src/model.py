
import torch
import torch.nn as nn
import logging, math
from layers import MLP, FuncToNodeSum

from torch.nn.utils.rnn import pad_sequence

_MEM_LOG = True    # set to False to disable GPU memory logging

def _mem(tag):
    if not _MEM_LOG or not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    alloc  = torch.cuda.memory_allocated()  / 1024**3
    peak   = torch.cuda.max_memory_allocated() / 1024**3
    reserv = torch.cuda.memory_reserved()   / 1024**3
    logging.info(f"[MEM] {tag:55s} alloc={alloc:.3f}GB  peak={peak:.3f}GB  reserved={reserv:.3f}GB")

def _concentration_from_mass(mass, dim):
    """1 - normalized entropy of a nonneg mass tensor along `dim`, in [0,1].

    Normalized by log(#nonzero active items) so it measures peakedness
    independent of count: one active item -> 1; uniform over k items -> 0;
    no grounding (total == 0) -> 0.

    Args:
        mass: float tensor, values >= 0.
        dim:  axis along which to compute concentration (int).
    Returns:
        concentration tensor with `dim` squeezed out.
    """
    total = mass.sum(dim=dim, keepdim=True)
    p = mass / total.clamp_min(1e-12)
    ent = -(p * p.clamp_min(1e-12).log()).sum(dim=dim)
    k = (mass > 0).sum(dim=dim)
    logk = k.float().clamp_min(1).log()
    norm_ent = torch.where(logk > 0, ent / logk.clamp_min(1e-12), torch.zeros_like(ent))
    conc = 1.0 - norm_ent
    return torch.where(total.squeeze(dim) > 0, conc, torch.zeros_like(conc))


class RulE(torch.nn.Module):
    def __init__(self, graph, p_norm, mlp_rule_dim, gamma_fact, gamma_rule, hidden_dim, device, dataset,
                 feature_name='density', use_per_relation=True):
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

        
        self.rule_to_entity = FuncToNodeSum(self.mlp_rule_dim)

        if "FB15k-237" in dataset or "wn18rr" in dataset or "YAGO3-10" in dataset:
            self.score_model = MLP(self.mlp_rule_dim, [128, 1]) 
        else:
            self.score_model = MLP(self.mlp_rule_dim, [1]) 

        self.bias = torch.nn.parameter.Parameter(torch.zeros(self.num_entities))

        # === Learnable per-relation beta for KGE/Rule balancing ===
        # beta * rule_score + (1-beta) * kge_score
        # sigmoid(0) = 0.5 -> equal weighting at init
        self.beta = nn.Parameter(torch.zeros(self.num_relations * 2))
        nn.init.constant_(self.beta, 0.0)

        # === Query-adaptive beta: conditions on a per-query feature ===
        # Full:          beta(h,r) = sigmoid(beta[r]     + beta_density * feature(h,r))
        # No-per-rel:    beta(h,r) = sigmoid(global_beta + beta_density * feature(h,r))
        self.beta_density = nn.Parameter(torch.zeros(1))

        # Global scalar intercept used when use_per_relation=False.
        # Replaces beta[r] with a single shared log-odds at the average feature value.
        self.global_beta = nn.Parameter(torch.zeros(1))

        # Whether to use per-relation intercept (beta[r]) or the global scalar.
        self.use_per_relation = use_per_relation

        # Which per-query feature the adaptive head conditions on:
        #   'density'   : fraction of candidate entities with >=1 rule fired.
        #   'num_rules' : fraction of this relation's rules that fired for the
        #                 (h, r) query (breadth of rule evidence).
        self.feature_name = feature_name

        # Optional standardization of the query feature before it enters the
        # sigmoid. When enabled, feature -> (feature - mean) / (std + eps) using
        # statistics computed once on the beta-training split. Stored as buffers
        # so they move with .to(device) and round-trip through state_dict.
        self.standardize_feature = False
        self.register_buffer('feature_mean', torch.zeros(1))
        self.register_buffer('feature_std', torch.ones(1))

        # Populated by forward():
        #   _last_ground_mask : bool [batch, num_entities], True where >=1 rule
        #                       fired (the 'density' signal, uncontaminated by
        #                       self.bias variance).
        #   _last_num_rules   : float [batch], fraction of the relation's rules
        #                       that fired for each query (the 'num_rules' signal).
        #   _last_conf_rules  : float [batch], fraction of the relation's total
        #                       rule-confidence MASS that fired for each query
        #                       (the 'conf_rules' signal). Requires
        #                       eval_compute_rule_confidence() to have run;
        #                       otherwise left None.
        #   _last_top_rule_conf : float [batch], max rule_confidence among rules
        #                       that fired for each query (the 'top_rule_confidence'
        #                       signal). 0 if no rule fired. Requires
        #                       eval_compute_rule_confidence(); else None.
        #   _last_rule_concentration : float [batch], 1 - normalized_entropy
        #                       over per-rule grounding mass (the 'rule_concentration'
        #                       signal). High = one rule dominates; low = many fire
        #                       evenly. 0 when no rule fired.
        #   _last_cand_concentration : float [batch], 1 - normalized_entropy
        #                       over summed grounding mass across candidate tails
        #                       (the 'cand_concentration' signal). High = groundings
        #                       concentrate on a few tails (discriminative); low =
        #                       spread across many tails (hub-dilution regime).
        #   _last_kge_max       : float [batch], max KGE score over filtered
        #                         candidates per query (the 'kge_max' signal).
        #                         NOT populated by forward(); must call
        #                         update_kge_stash(kge_score, flag) after
        #                         compute_g_KGE(). None until first call.
        #   _last_kge_entropy   : float [batch], Shannon entropy of softmax over
        #                         filtered KGE scores (the 'kge_entropy' signal).
        #                         High = flat KGE distribution (uncertain); low =
        #                         peaked (confident). NOT populated by forward();
        #                         computed together with _last_kge_max inside
        #                         update_kge_stash(). None until first call.
        self._last_ground_mask = None
        self._last_num_rules = None
        self._last_conf_rules = None
        self._last_top_rule_conf = None
        self._last_rule_concentration = None
        self._last_cand_concentration = None
        self._last_kge_max = None
        self._last_kge_entropy = None

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


        self.mlp_feature = nn.Parameter(torch.zeros(self.num_rules, self.mlp_rule_dim))
        
        nn.init.kaiming_uniform_(self.mlp_feature, a=math.sqrt(5), mode="fan_in")

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
    

    def forward(self, all_h, all_r, edges_to_remove):
        query_r = all_r[0].item()
        assert (all_r != query_r).sum() == 0
        device = all_r.device

        if device.type == "cuda":
            self.rule_features = self.rule_features.cuda(device)

        rule_index = list()
        rule_count = list()

        _mem(f"forward START  B={all_h.size(0)} r={query_r}")

        num_rules_total = len(self.relation2rules[query_r])
        # Per-query count of distinct rules that fired (grounded to >=1 entity).
        rules_fired = torch.zeros(all_h.size(0), device=device)

        # Confidence-mass weighting -- only when per-rule confidence has been
        # precomputed (eval_compute_rule_confidence). conf_fired accumulates the
        # confidence of fired rules per query; conf_total is the relation's total
        # confidence mass (the normalizer, constant across queries of this batch).
        have_conf = getattr(self, 'rule_confidence', None) is not None
        conf_fired = torch.zeros(all_h.size(0), device=device) if have_conf else None
        conf_total = torch.zeros((), device=device) if have_conf else None

        # Per-rule grounding mass [B] per rule, stacked after the loop to [B, R]
        # for rule_concentration. Top-confidence of any fired rule per query.
        per_rule_mass = []
        top_conf = torch.zeros(all_h.size(0), device=device) if have_conf else None

        mask = torch.zeros(all_h.size(0), self.graph.entity_size, device=device)
        for index, (r_head, r_body) in self.relation2rules[query_r]:

            assert r_head == query_r

            count = self.graph.grounding(all_h, r_head, r_body, edges_to_remove).float()
            
            mask += count
            # This rule "fired" for query b iff it grounded to >=1 candidate tail
            # for head h_b. Reduce over the entity axis -> [batch].
            mass = count.sum(dim=-1)
            fired = (mass > 0).float()
            rules_fired += fired
            per_rule_mass.append(mass)

            if have_conf:
                c = self.rule_confidence[index]
                conf_total = conf_total + c
                conf_fired = conf_fired + c * fired
                # Maximum confidence among rules that fired; valid because
                # rule_confidence > 0 always (sigmoid output).
                top_conf = torch.maximum(top_conf, c * fired)

            rule_index.append(index)
            rule_count.append(count)

        _mem(f"after grounding loop  num_rules={len(rule_index)}")

        # Fraction of this relation's rules that fired per query (in [0, 1]) so a
        # single shared beta_density slope stays comparable across relations.
        self._last_num_rules = rules_fired / max(num_rules_total, 1)

        # Confidence-weighted analogue: fraction of the relation's total
        # rule-confidence mass that fired for each query (in [0, 1]). Upweights
        # queries where the rules that fired are the ones RulE trusts most.
        if have_conf:
            self._last_conf_rules = conf_fired / conf_total.clamp_min(1e-12)
            self._last_top_rule_conf = top_conf
        else:
            self._last_conf_rules = None
            self._last_top_rule_conf = None

        # Rule-axis concentration: 1 - normalized_entropy over per-rule grounding
        # mass [B, R]. High = one rule dominates; low = many rules fire evenly.
        if per_rule_mass:
            rule_mass_t = torch.stack(per_rule_mass, dim=1)  # [B, R]
            self._last_rule_concentration = _concentration_from_mass(rule_mass_t, dim=1)
        else:
            self._last_rule_concentration = torch.zeros(all_h.size(0), device=device)

        if mask.sum().item() == 0:
            # No rule fired for any query in this batch. Stash an all-False
            # grounding mask so any consumer of self._last_ground_mask
            # (compute_adaptive_beta, the per-bucket density analysis) sees
            # density = 0 for every query, which is the correct value here.
            self._last_ground_mask = torch.zeros_like(mask, dtype=torch.bool)
            self._last_cand_concentration = torch.zeros(all_h.size(0), device=device)
            return mask + self.bias.unsqueeze(0), (1 - mask).bool()

        # Snapshot the pre-bias grounding mask before `mask` is overwritten
        # below. True iff at least one rule fired on that (query, entity) pair.
        # This is the actual "rule-grounding density" signal -- the post-bias
        # score variance is dominated by self.bias and is not a usable proxy.
        self._last_ground_mask = (mask > 0)

        # Candidate-axis concentration: 1 - normalized_entropy of summed grounding
        # mass across candidate tails [B, E]. High = groundings concentrate on a
        # few tails (discriminative); low = spread across many tails (hub-dilution).
        # Computed here before `mask` is overwritten to bool at the end.
        self._last_cand_concentration = _concentration_from_mass(mask, dim=1)

        candidate_set = torch.nonzero(mask.view(-1), as_tuple=True)[0]
        _mem(f"after candidate_set  C={candidate_set.size(0)}")

        rule_index = torch.tensor(rule_index, dtype=torch.long, device=device)
        rule_count = torch.stack(rule_count, dim=0)
        _mem(f"after stack rule_count  shape={list(rule_count.shape)}")

        rule_count = rule_count.reshape(rule_index.size(0), -1)[:, candidate_set]
        _mem(f"after candidate filter  shape={list(rule_count.shape)}")

        rule_emb = self.rules_weight_emb[rule_index]
        mlp_feature = self.mlp_feature[rule_index]

        output = self.rule_to_entity(rule_count, rule_emb, mlp_feature)
        _mem(f"after rule_to_entity  out={list(output.shape)}")

        feature = output
        output = self.score_model(feature).squeeze(-1)
        _mem(f"after score_model")

        score = torch.zeros(all_h.size(0) * self.graph.entity_size, device=device)
        score.scatter_(0, candidate_set, output)
        score = score.view(all_h.size(0), self.graph.entity_size)
        score = score + self.bias.unsqueeze(0)

        mask = torch.ones_like(mask).bool()
        _mem(f"forward END")

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

    def eval_compute_rule_confidence(self, device):
        '''
        Precompute a per-rule scalar "confidence" in (0, 1), used to weight the
        distinct-rule-count feature (conf_rules). The confidence is RulE's own
        rule-plausibility score from add_ruleE -- gamma_rule - ||body - head||,
        higher = the rule embedding satisfies the relation algebra better --
        squashed through a sigmoid so it is a bounded, positive weight.

        Stored as self.rule_confidence, aligned with the rule index used in
        forward() (rule_features[i] <-> rule id i <-> rules_weight_emb[i]). Call
        once after load (mirrors eval_compute_rule_weight); forward() only emits
        self._last_conf_rules when this has run.
        '''
        batch = 128
        self.rule_masks = self.rule_masks.to(device)
        self.rule_features = self.rule_features.to(device)
        split_num = max(self.rule_features.size(0) // batch, 1)
        rule_batches = torch.split(self.rule_features, split_num, 0)
        rule_mask_batches = torch.split(self.rule_masks, split_num, 0)
        plausibility = list()

        with torch.no_grad():
            for rules, rules_mask in zip(rule_batches, rule_mask_batches):
                dist, _ = self.add_ruleE(rules.unsqueeze(1), rules_mask)
                plausibility.append(dist.squeeze(1))

        self.rule_plausibility = torch.cat(plausibility)          # [num_rules]
        self.rule_confidence = torch.sigmoid(self.rule_plausibility)  # (0, 1)

    def update_kge_stash(self, kge_score, flag):
        """Stash per-query KGE summary statistics for use by get_query_feature().

        Call after compute_g_KGE() and before compute_adaptive_beta(). Safe to
        call unconditionally: always populates _last_kge_max and _last_kge_entropy
        regardless of feature_name so callers need not branch.

        Args:
            kge_score: [batch, num_entities] float tensor of KGE scores.
            flag:      [batch, num_entities] bool, True for filtered candidates
                       (entities not known to be positive for this query).
        """
        # Fill non-candidates with -inf so they are ignored by max() and softmax().
        masked = kge_score.masked_fill(~flag, float('-inf'))
        self._last_kge_max = masked.max(dim=-1).values  # [batch]

        # Shannon entropy of softmax over filtered candidates.
        # softmax(-inf) = 0, so non-candidates contribute nothing and the
        # shift-invariance of softmax means the raw RotatE scale is irrelevant.
        # For queries with no filtered candidates (flag all-False) softmax is
        # undefined; set entropy to 0 for those queries.
        p = torch.softmax(masked, dim=-1)
        ent = -(p * torch.log(p.clamp_min(1e-12))).sum(dim=-1)
        self._last_kge_entropy = torch.where(
            flag.any(dim=-1), ent, torch.zeros_like(ent)
        )

    def get_query_feature(self):
        """Raw per-query feature [batch] selected by self.feature_name.

        'density'           : fraction of candidate entities with >=1 rule fired
                              (read from _last_ground_mask). The post-bias rule
                              score is NOT a usable proxy -- its variance is
                              dominated by self.bias.
        'num_rules'         : fraction of this relation's rules that fired for
                              the (h, r) query (read from _last_num_rules) -- a
                              breadth-of-evidence signal.
        'kge_max'           : max KGE score over filtered candidates (read from
                              _last_kge_max). Populated by update_kge_stash()
                              after compute_g_KGE(). Needs --standardize_feature
                              because the raw value is dataset/relation-scale-
                              dependent.
        'kge_entropy'       : Shannon entropy of softmax over filtered KGE scores
                              (read from _last_kge_entropy). High = flat/uncertain
                              KGE distribution; low = peaked/confident. Populated
                              by update_kge_stash(); needs --standardize_feature.
        'top_rule_confidence': max RulE plausibility (sigmoid of rule embedding
                              distance) among rules that fired for this query; 0
                              if no rule fired (read from _last_top_rule_conf).
                              Requires eval_compute_rule_confidence() to have run
                              before the eval loop.

        forward() must have been called first. For kge_max / kge_entropy,
        update_kge_stash() must also have been called after compute_g_KGE().
        For top_rule_confidence, eval_compute_rule_confidence() must have run.
        """
        if self.feature_name == 'kge_max':
            if self._last_kge_max is None:
                raise RuntimeError(
                    "get_query_feature('kge_max') requires update_kge_stash() "
                    "to have been called after compute_g_KGE(); "
                    "self._last_kge_max is None."
                )
            return self._last_kge_max
        if self.feature_name == 'kge_entropy':
            if self._last_kge_entropy is None:
                raise RuntimeError(
                    "get_query_feature('kge_entropy') requires update_kge_stash() "
                    "to have been called after compute_g_KGE(); "
                    "self._last_kge_entropy is None."
                )
            return self._last_kge_entropy
        if self.feature_name == 'top_rule_confidence':
            if self._last_top_rule_conf is None:
                raise RuntimeError(
                    "get_query_feature('top_rule_confidence') requires "
                    "eval_compute_rule_confidence() to have run before the eval "
                    "loop; self._last_top_rule_conf is None."
                )
            return self._last_top_rule_conf
        if self.feature_name == 'num_rules':
            if self._last_num_rules is None:
                raise RuntimeError(
                    "get_query_feature('num_rules') requires forward() first; "
                    "self._last_num_rules is None."
                )
            return self._last_num_rules
        if self._last_ground_mask is None:
            raise RuntimeError(
                "get_query_feature('density') requires forward() first; "
                "self._last_ground_mask is None."
            )
        return self._last_ground_mask.float().mean(dim=-1)   # [batch]

    def compute_adaptive_beta(self, rule_score, all_r):
        """Query-adaptive beta = sigmoid(beta[r] + beta_density * feature).

        The feature is selected by self.feature_name ('density' or 'num_rules')
        and read from the stashes populated by forward(). When
        self.standardize_feature is True the feature is z-scored with
        feature_mean / feature_std (computed once on the beta-training split) so
        beta[r] is the log-odds at the *average* query and the slope
        beta_density is well-conditioned and comparable across relations.

        Args:
            rule_score: [batch, num_entities] (kept for API stability, unused).
            all_r:      [batch] relation indices.

        Returns:
            beta:    [batch, 1] adaptive beta in (0, 1).
            feature: [batch] the RAW (un-standardized) feature per query, for
                     interpretability / bucketing.
        """
        raw = self.get_query_feature()                          # [batch]
        if self.standardize_feature:
            feat = (raw - self.feature_mean) / (self.feature_std + 1e-6)
        else:
            feat = raw

        if self.use_per_relation:
            intercept = self.beta[all_r]                        # [batch]
        else:
            intercept = self.global_beta.expand(all_r.shape[0])  # [batch], same scalar
        logit = intercept + self.beta_density * feat
        beta = torch.sigmoid(logit).unsqueeze(-1)
        return beta, raw
