
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
    
    Also provides a confidence projection that maps rule embeddings to
    scalar quality scores.
    """

    def __init__(self, entity_dim, relation_dim, attn_dim=128, negative_slope=0.2):
        super(GroundingGAT, self).__init__()

        if attn_dim is None:
            attn_dim = 128

        self.attn_dim = attn_dim
        self.negative_slope = negative_slope

        self.W_src = nn.Linear(entity_dim, attn_dim, bias=False)
        self.W_dst = nn.Linear(entity_dim, attn_dim, bias=False)
        self.W_rel = nn.Linear(relation_dim, attn_dim, bias=False)
        self.attn_vec = nn.Linear(attn_dim, 1, bias=False)

        self.conf_proj = nn.Linear(relation_dim, 1)

        # Accumulator for the attention-entropy regulariser. Reset to None at the
        # start of every forward pass; populated inside compute_edge_attention.
        # Holds Σ a*log(a) summed across all edges of all hops in this batch.
        # Adding lambda * this to the loss with lambda > 0 rewards uniform
        # attention (Σ a*log a is most negative when uniform, ~0 when peaked),
        # countering the GAT attention-collapse pathology.
        self.attn_entropy_penalty = None

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
        nn.init.xavier_uniform_(self.W_dst.weight)
        nn.init.xavier_uniform_(self.W_rel.weight)
        nn.init.xavier_uniform_(self.attn_vec.weight)
        nn.init.xavier_uniform_(self.conf_proj.weight)
        nn.init.zeros_(self.conf_proj.bias)

    def compute_edge_attention(self, entity_emb, node_in, node_out, rel_emb, num_nodes):
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

        Returns:
            edge_attn: [num_edges]  attention weight per edge, sums to 1 per target
        """
        # Project all entities to attn_dim first, then index by edge endpoints.
        # This avoids creating [num_edges, entity_dim] copies that would be stored
        # for the backward pass; backward of W_src/W_dst uses entity_emb which is
        # already resident in GPU memory.
        all_src_proj = F.linear(entity_emb, self.W_src.weight)  # [num_entities, attn_dim]
        all_dst_proj = F.linear(entity_emb, self.W_dst.weight)  # [num_entities, attn_dim]
        src_proj = all_src_proj[node_in]   # [num_edges, attn_dim]
        dst_proj = all_dst_proj[node_out]  # [num_edges, attn_dim]

        proj = src_proj + dst_proj + self.W_rel(rel_emb)
        proj = F.leaky_relu(proj, negative_slope=self.negative_slope)
        edge_score = self.attn_vec(proj).squeeze(-1)            # [num_edges]
        edge_attn = scatter_softmax(edge_score, node_out, dim=0, dim_size=num_nodes)
        self.last_attn = edge_attn.detach()

        # Accumulate Σ a*log(a) across all hops in this forward pass. Stays in
        # the graph so the regulariser flows gradients into W_src/W_dst/W_rel/
        # attn_vec when the trainer adds lambda * attn_entropy_penalty to the loss.
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
