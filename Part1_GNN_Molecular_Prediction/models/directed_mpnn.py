import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import scatter


class DirectedMPNNRegressor(nn.Module):
    def __init__(
        self,
        node_dim,
        edge_dim,
        global_feat_dim,
        hidden_dim=128,
        num_layers=3,
        dropout=0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.init_edge = nn.Linear(node_dim + edge_dim, hidden_dim)
        self.edge_updates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for _ in range(num_layers)
        ])

        self.global_proj = (
            nn.Linear(global_feat_dim, hidden_dim) if global_feat_dim > 0 else None
        )

        combined_dim = hidden_dim + (hidden_dim if self.global_proj is not None else 0)
        self.predictor = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, edge_index, edge_attr, batch, global_feats=None):
        num_nodes = x.size(0)
        src_nodes = edge_index[0]
        edge_state = self.init_edge(torch.cat([x[src_nodes], edge_attr], dim=-1))

        for layer in range(self.num_layers):
            node_messages = scatter(edge_state, edge_index[1], dim=0, dim_size=num_nodes, reduce='add')
            src_messages = node_messages[src_nodes]
            edge_state = self.edge_updates[layer](torch.cat([edge_state, src_messages], dim=-1))

        node_repr = scatter(edge_state, edge_index[1], dim=0, dim_size=num_nodes, reduce='add')
        graph_repr = global_add_pool(node_repr, batch)

        if self.global_proj is not None:
            if global_feats is None:
                batch_size = batch.max().item() + 1
                global_repr = torch.zeros(batch_size, self.global_proj.out_features, device=x.device)
            else:
                global_repr = self.global_proj(global_feats)
            combined = torch.cat([graph_repr, global_repr], dim=-1)
        else:
            combined = graph_repr

        return self.predictor(combined)


def create_directed_mpnn_model(
    num_node_features,
    num_edge_features,
    global_feat_dim,
    device,
    hidden_channels=128,
    num_layers=3,
    dropout=0.2,
):
    model = DirectedMPNNRegressor(
        node_dim=num_node_features,
        edge_dim=num_edge_features,
        global_feat_dim=global_feat_dim,
        hidden_dim=hidden_channels,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    return model
