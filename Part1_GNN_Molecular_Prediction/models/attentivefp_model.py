import torch
import torch.nn as nn
from torch_geometric.nn import AttentiveFP


class AttentiveFPRegressor(nn.Module):
    def __init__(
        self,
        num_node_features,
        num_edge_features,
        global_feat_dim=0,
        hidden_channels=128,
        num_layers=3,
        num_timesteps=2,
        dropout=0.2,
    ):
        super().__init__()
        self.gnn = AttentiveFP(
            in_channels=num_node_features,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            edge_dim=num_edge_features,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
            dropout=dropout,
        )

        self.global_proj = (
            nn.Linear(global_feat_dim, hidden_channels) if global_feat_dim > 0 else None
        )

        combined_dim = hidden_channels + (hidden_channels if self.global_proj is not None else 0)
        self.predictor = nn.Sequential(
            nn.Linear(combined_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, x, edge_index, edge_attr, batch, global_feats=None):
        mol_representation = self.gnn(x, edge_index, edge_attr, batch)

        if self.global_proj is not None:
            if global_feats is None:
                batch_size = batch.max().item() + 1
                global_representation = torch.zeros(batch_size, self.global_proj.out_features, device=x.device)
            else:
                global_representation = self.global_proj(global_feats)
            combined = torch.cat([mol_representation, global_representation], dim=-1)
        else:
            combined = mol_representation

        return self.predictor(combined)


def create_attentivefp_model(
    num_node_features,
    num_edge_features,
    global_feat_dim,
    device,
    hidden_channels=128,
    num_layers=3,
    dropout=0.2,
    num_timesteps=2,
):
    model = AttentiveFPRegressor(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        global_feat_dim=global_feat_dim,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        num_timesteps=num_timesteps,
        dropout=dropout,
    ).to(device)
    return model