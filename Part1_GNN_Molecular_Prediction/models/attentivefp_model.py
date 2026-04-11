import torch
from torch_geometric.nn import AttentiveFP

def create_attentivefp_model(num_node_features, num_edge_features, device):
    """
    Creates and returns the AttentiveFP model based on professor's guidelines.
    """
    model = AttentiveFP(
        in_channels=num_node_features,
        hidden_channels=128,          # Increased for better learning
        out_channels=1,               # Output is 1 (Solubility prediction)
        edge_dim=num_edge_features,
        num_layers=3,                 # Advanced deep layers
        num_timesteps=2,
        dropout=0.2                   # Dropout to prevent overfitting
    ).to(device)
    
    return model

if __name__ == "__main__":
    print("Model module is ready to be imported!")