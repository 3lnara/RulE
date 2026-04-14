
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        
    
    def forward(self, A_fn, x_f, mlp_rule_feature, chunk_size=512):
        # A_fn: [num_rules, num_candidates]
        # x_f:  [num_rules, rule_dim]
        # mlp_rule_feature: [num_rules, mlp_rule_dim]
        #
        # The naive broadcast (message * weight) is [num_candidates, num_rules, rule_dim]
        # which OOMs on large graphs (e.g. WN18RR with 40k entities).
        # We chunk over candidates to keep peak memory at chunk_size * num_rules * rule_dim.

        weight = torch.transpose(A_fn, 0, 1).unsqueeze(-1)  # [C, R, 1]
        message = x_f.unsqueeze(0)                          # [1, R, D]

        num_candidates = weight.size(0)
        outputs = []
        for start in range(0, num_candidates, chunk_size):
            w_chunk = weight[start: start + chunk_size]              # [c, R, 1]
            feat = torch.transpose((message * w_chunk), 1, 2)        # [c, D, R]
            wf = torch.matmul(feat, mlp_rule_feature)                # [c, D, mlp_dim]
            wf = self.layer_norm(wf)
            wf = torch.relu(wf)
            outputs.append(wf.mean(1))                               # [c, mlp_dim]

        return torch.cat(outputs, dim=0)                             # [C, mlp_dim]



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
