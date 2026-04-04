
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

    def __init__(self, entity_dim, relation_dim, attn_dim=None, negative_slope=0.2):
        super(GroundingGAT, self).__init__()

        if attn_dim is None:
            attn_dim = relation_dim

        self.attn_dim = attn_dim
        self.negative_slope = negative_slope

        self.W_src = nn.Linear(entity_dim, attn_dim, bias=False)
        self.W_dst = nn.Linear(entity_dim, attn_dim, bias=False)
        self.W_rel = nn.Linear(relation_dim, attn_dim, bias=False)
        self.attn_vec = nn.Linear(attn_dim, 1, bias=False)

        self.conf_proj = nn.Linear(relation_dim, 1)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W_src.weight)
        nn.init.xavier_uniform_(self.W_dst.weight)
        nn.init.xavier_uniform_(self.W_rel.weight)
        nn.init.xavier_uniform_(self.attn_vec.weight)
        nn.init.xavier_uniform_(self.conf_proj.weight)
        nn.init.zeros_(self.conf_proj.bias)

    def compute_edge_attention(self, src_emb, dst_emb, rel_emb, node_out, num_nodes):
        """
        Compute GATv2 attention for each edge, normalized per target node.

        Args:
            src_emb:  [num_edges, entity_dim]
            dst_emb:  [num_edges, entity_dim]
            rel_emb:  [relation_dim]  (single vector, broadcast over edges)
            node_out: [num_edges]  target node indices (for scatter_softmax)
            num_nodes: int  total number of nodes

        Returns:
            edge_attn: [num_edges]  attention weight per edge, sums to 1 per target
        """
        proj = self.W_src(src_emb) + self.W_dst(dst_emb) + self.W_rel(rel_emb)
        proj = F.leaky_relu(proj, negative_slope=self.negative_slope)
        edge_score = self.attn_vec(proj).squeeze(-1)            # [num_edges]
        edge_attn = scatter_softmax(edge_score, node_out, dim=0, dim_size=num_nodes)
        return edge_attn

    def compute_confidence(self, rule_emb):
        """
        Map a rule's embedding vector to a scalar confidence in [0, 1].

        Args:
            rule_emb: [hidden_dim] or [num_rules, hidden_dim]

        Returns:
            confidence: same leading dims, scalar per rule
        """
        return torch.sigmoid(self.conf_proj(rule_emb)).squeeze(-1)
