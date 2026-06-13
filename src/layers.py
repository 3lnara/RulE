
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_softmax


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, short_cut=False, batch_norm=False, activation="relu", dropout=0):
        super(MLP, self).__init__()

        self.dims = [input_dim] + hidden_dims
        self.short_cut = short_cut

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))
        if batch_norm:
            self.batch_norms = nn.ModuleList()
            for i in range(len(self.dims) - 2):
                self.batch_norms.append(nn.BatchNorm1d(self.dims[i + 1]))
        else:
            self.batch_norms = None

    def forward(self, input):
        layer_input = input

        for i, layer in enumerate(self.layers):
            hidden = layer(layer_input)
            if i < len(self.layers) - 1:
                if self.batch_norms:
                    x = hidden.flatten(0, -2)
                    hidden = self.batch_norms[i](x).view_as(hidden)
                hidden = self.activation(hidden)
                if self.dropout:
                    hidden = self.dropout(hidden)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            layer_input = hidden

        return hidden

class FuncToNodeSum(nn.Module):
    def __init__(self, vector_dim):
        super(FuncToNodeSum, self).__init__()

        self.vector_dim = vector_dim
        self.layer_norm = nn.LayerNorm(self.vector_dim)
        self.add_model = MLP(self.vector_dim, [self.vector_dim])
        # for param in self.add_model.parameters():
        #     param.requires_grad = False
        
    
    def forward(self, A_fn, x_f, mlp_rule_feature):
        
        weight = torch.transpose(A_fn, 0, 1).unsqueeze(-1)
        message = x_f.unsqueeze(0)

        feature = torch.transpose((message * weight), 1, 2)
        weighted_features = torch.matmul(feature, mlp_rule_feature)
        weighted_features_norm = self.layer_norm(weighted_features)
        weighted_features_relu = torch.relu(weighted_features_norm)
        output = weighted_features_relu.mean(1)
        
        return output


    # def forward(self, A_fn, x_f):
        
    #     # batch_size = b_n.max().item() + 1

        
    #     weight = torch.transpose(A_fn, 0, 1).unsqueeze(-1)
    #     message = x_f.unsqueeze(0)


    #     features = (message * weight).sum(1)
    #     # features = (message * weight).mean(1)

    #     # features = (message * weight).max(1)[0]

    #     output = self.add_model(features)
    #     output = self.layer_norm(output)
    #     output = torch.relu(output)

    #     return output


class GroundingGAT(nn.Module):
    """
    GATv2-style edge attention for rule grounding propagation.

    At each hop in a rule body, computes per-edge attention weights from
    source/target entity embeddings and the relation embedding. These weights
    modulate message passing so that the final grounding score at each
    candidate entity is an attention-weighted sum over grounding paths.

    Variants (selected via the `variant` constructor argument):
      * 'baseline'  -- GATv2: proj = W_src(x_in) + W_dst(x_out) + W_rel(rel)
                       scored by a shared attn_vec.
      * 'no_dst'    -- drop W_dst (provably redundant under per-target softmax;
                       only its LeakyReLU-kink contribution survives in the
                       baseline). Keeps the shared attn_vec.
      * 'rule_attn' -- drop W_dst AND replace the shared attn_vec with a
                       per-rule scoring vector produced by
                       attn_from_rule(rule_emb). The rule embedding then
                       controls which attn_dim features matter, so two rules
                       see different per-edge rankings even on the same edge
                       set.

    Also provides a confidence projection that maps rule embeddings to
    scalar quality scores.
    """

    _VALID_VARIANTS = ('baseline', 'no_dst', 'rule_attn')

    def __init__(self, entity_dim, relation_dim, attn_dim=128, rule_dim=None,
                 variant='baseline', negative_slope=0.2, checkpoint_attention=False):
        super(GroundingGAT, self).__init__()

        if attn_dim is None:
            attn_dim = 128
        if variant not in self._VALID_VARIANTS:
            raise ValueError(
                f"Unknown GAT variant '{variant}'. "
                f"Expected one of {self._VALID_VARIANTS}."
            )
        if variant == 'rule_attn' and rule_dim is None:
            raise ValueError("variant='rule_attn' requires rule_dim to be set.")

        self.variant = variant
        self.attn_dim = attn_dim
        self.negative_slope = negative_slope
        # When True, compute_edge_attention is run inside a gradient checkpoint
        # (see GraphData.propagate_with_attention) so its [num_edges, attn_dim]
        # activations are freed after the forward and recomputed in backward.
        # This lets large graphs (e.g. WN18RR) fit at attn_dim=128 on a 16 GB GPU.
        # It is incompatible with the attention-entropy regulariser, so the
        # entropy accumulation below is skipped whenever this flag is set.
        self.checkpoint_attention = checkpoint_attention

        self.W_src = nn.Linear(entity_dim, attn_dim, bias=False)
        # W_dst only exists in the baseline variant.
        if variant == 'baseline':
            self.W_dst = nn.Linear(entity_dim, attn_dim, bias=False)
        self.W_rel = nn.Linear(relation_dim, attn_dim, bias=False)

        if variant == 'rule_attn':
            # Per-rule attention vector: a = attn_from_rule(rule_emb) gives
            # each rule its own scoring direction in attn_dim space.
            self.attn_from_rule = nn.Linear(rule_dim, attn_dim, bias=False)
        else:
            self.attn_vec = nn.Linear(attn_dim, 1, bias=False)

        self.conf_proj = nn.Linear(relation_dim, 1)

        # Accumulator for the attention-entropy regulariser. Reset to None at the
        # start of every forward pass; populated inside compute_edge_attention.
        # Holds Σ a*log(a) summed across all edges of all hops in this batch.
        # Adding lambda * this to the loss with lambda > 0 rewards uniform
        # attention (Σ a*log a is most negative when uniform, ~0 when peaked),
        # countering the GAT attention-collapse pathology.
        self.attn_entropy_penalty = None

        # Per-forward cache of the shared entity->attn_dim projections. During
        # grounding, entity_emb is frozen and W_src/W_dst are shared across every
        # rule and hop, so F.linear(entity_emb, W_src.weight) is identical on
        # every compute_edge_attention call within a forward pass. The GAT loops
        # over up to ~400 rules x up to 5 hops per batch, so without this cache
        # the [num_entities, attn_dim] matmul (40,943 x 1,000 -> 128 on WN18RR)
        # is recomputed ~1,000 times per batch. precompute_entity_projections()
        # populates these once; compute_edge_attention() reuses them, making the
        # result bit-identical while running the matmul exactly once.
        self._cached_all_src_proj = None
        self._cached_all_dst_proj = None

        self._reset_parameters()
    def set_frozen_confidences(self, confidence_tensor):
        """
        Store precomputed RulE confidences as a non-trainable buffer.
        Should be called once after eval_compute_rule_weight runs in stage two.
        
        Args:
            confidence_tensor: [num_rules] scalar confidence per rule.
                            Should be sigmoid(add_ruleE scores) computed from
                            the pre-trained embeddings (see GroundTrainer.train).
        """
        self.register_buffer('frozen_confidences', confidence_tensor.clone().detach())
        self.register_buffer('use_frozen_confidence', torch.ones(1, dtype=torch.bool))
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W_src.weight)
        if self.variant == 'baseline':
            nn.init.xavier_uniform_(self.W_dst.weight)
        nn.init.xavier_uniform_(self.W_rel.weight)
        if self.variant == 'rule_attn':
            nn.init.xavier_uniform_(self.attn_from_rule.weight)
        else:
            nn.init.xavier_uniform_(self.attn_vec.weight)
        nn.init.xavier_uniform_(self.conf_proj.weight)
        nn.init.zeros_(self.conf_proj.bias)

    def precompute_entity_projections(self, entity_emb):
        """
        Precompute the shared entity->attn_dim projections once per forward.

        entity_emb is frozen during grounding and W_src/W_dst are shared across
        every rule and every hop, so F.linear(entity_emb, W_*.weight) is the same
        on every compute_edge_attention call within a forward. Caching it turns
        the >1,000-per-batch [num_entities, attn_dim] matmul (e.g. 40,943 x 1,000
        -> 128 on WN18RR) into a single matmul. Results are bit-identical and the
        cached tensors stay in the autograd graph (grad flows to W_src/W_dst), so
        gradients remain exact. compute_edge_attention reads these via `self`,
        including under gradient checkpointing -- the same access pattern already
        used for self.W_src.weight there -- so the matmul is also not recomputed
        in backward.
        """
        self._cached_all_src_proj = F.linear(entity_emb, self.W_src.weight)
        if self.variant == 'baseline':
            self._cached_all_dst_proj = F.linear(entity_emb, self.W_dst.weight)
        else:
            self._cached_all_dst_proj = None

    def clear_entity_projection_cache(self):
        """Drop the cached projections (release the graph they hold)."""
        self._cached_all_src_proj = None
        self._cached_all_dst_proj = None

    def compute_edge_attention(self, entity_emb, node_in, node_out, rel_emb, num_nodes,
                               rule_emb=None):
        """
        Compute GATv2 attention for each edge, normalized per target node.

        Projects all entity embeddings to attn_dim first (cheap: num_entities rows),
        then indexes by edge, avoiding large [num_edges, entity_dim] materialization.

        Args:
            entity_emb: [num_entities, entity_dim]
            node_in:    [num_edges]  source node indices
            node_out:   [num_edges]  target node indices (for scatter_softmax)
            rel_emb:    [relation_dim]  (single vector, broadcast over edges)
            num_nodes:  int  total number of nodes
            rule_emb:   [rule_dim] or None. Required iff variant='rule_attn'.

        Returns:
            edge_attn: [num_edges]  attention weight per edge, sums to 1 per target
        """
        # Project all entities to attn_dim first, then index by edge endpoints.
        # This avoids creating [num_edges, entity_dim] copies that would be stored
        # for the backward pass; backward of W_src/W_dst uses entity_emb which is
        # already resident in GPU memory.
        #
        # The full [num_entities, attn_dim] projection is identical for every rule
        # and hop in a forward (entity_emb frozen, W_src/W_dst shared), so reuse
        # the per-forward cache populated by precompute_entity_projections() when
        # present. The fallback recomputes it for standalone calls.
        if self._cached_all_src_proj is not None:
            all_src_proj = self._cached_all_src_proj
        else:
            all_src_proj = F.linear(entity_emb, self.W_src.weight)  # [num_entities, attn_dim]
        src_proj = all_src_proj[node_in]   # [num_edges, attn_dim]

        if self.variant == 'baseline':
            if self._cached_all_dst_proj is not None:
                all_dst_proj = self._cached_all_dst_proj
            else:
                all_dst_proj = F.linear(entity_emb, self.W_dst.weight)  # [num_entities, attn_dim]
            dst_proj = all_dst_proj[node_out]  # [num_edges, attn_dim]
            proj = src_proj + dst_proj + self.W_rel(rel_emb)
        else:
            # no_dst / rule_attn: drop the dst term entirely. Under per-target
            # softmax it acts as a per-group constant whose only signal comes
            # through the LeakyReLU kink, which is empirically near-noise.
            proj = src_proj + self.W_rel(rel_emb)

        proj = F.leaky_relu(proj, negative_slope=self.negative_slope)

        if self.variant == 'rule_attn':
            if rule_emb is None:
                raise ValueError("variant='rule_attn' requires rule_emb at attention time.")
            # Per-rule scoring vector. proj: [E, attn_dim]; a: [attn_dim];
            # broadcasting -> [E, attn_dim], sum over attn_dim -> [E].
            a = self.attn_from_rule(rule_emb)
            edge_score = (proj * a).sum(-1)                     # [num_edges]
        else:
            edge_score = self.attn_vec(proj).squeeze(-1)        # [num_edges]

        edge_attn = scatter_softmax(edge_score, node_out, dim=0, dim_size=num_nodes)
        self.last_attn = edge_attn.detach()

        # Accumulate Σ a*log(a) across all hops in this forward pass. Stays in
        # the graph so the regulariser flows gradients into W_src/W_dst/W_rel/
        # attn_vec when the trainer adds lambda * attn_entropy_penalty to the loss.
        # Skip when checkpointing: the entropy term would reference edge_attn's
        # graph, which the checkpoint is meant to free, and the accumulation
        # would also double-count when this function re-runs during backward.
        if not self.checkpoint_attention:
            eps = 1e-9
            penalty = (edge_attn * (edge_attn + eps).log()).sum()
            if self.attn_entropy_penalty is None:
                self.attn_entropy_penalty = penalty
            else:
                self.attn_entropy_penalty = self.attn_entropy_penalty + penalty

        return edge_attn

    def compute_confidence(self, rule_index=None, rule_emb=None):
        if getattr(self, 'use_frozen_confidence', False):
            # Look up the precomputed RulE confidence for this rule.
            # This mirrors the original RulE pattern where rules_weight_emb
            # is indexed by rule_index inside model.forward.
            return self.frozen_confidences[rule_index]
        else:
            # Original behaviour: learnable linear projection of rule embedding.
            return torch.sigmoid(self.conf_proj(rule_emb)).squeeze(-1)
