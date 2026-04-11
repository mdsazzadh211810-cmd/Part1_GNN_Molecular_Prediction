import torch
from torch_geometric.nn import AttentiveFP

def create_attentivefp_model(num_node_features, num_edge_features, device, hidden_channels=128, num_layers=3, dropout=0.2):
    """
    Creates and returns the AttentiveFP model with dynamic hyperparameters.
    """
    model = AttentiveFP(
        in_channels=num_node_features,
        hidden_channels=hidden_channels,
        out_channels=1,
        edge_dim=num_edge_features,
        num_layers=num_layers,
        num_timesteps=2,
        dropout=dropout
    ).to(device)
    
    return model