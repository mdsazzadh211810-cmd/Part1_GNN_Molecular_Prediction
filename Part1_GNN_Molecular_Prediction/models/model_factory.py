from .attentivefp_model import create_attentivefp_model
from .directed_mpnn import create_directed_mpnn_model


def create_model(
    model_name,
    num_node_features,
    num_edge_features,
    global_feat_dim,
    device,
    hidden_channels=128,
    num_layers=3,
    dropout=0.2,
    num_timesteps=2,
):
    model_name = model_name.lower()
    if model_name == 'attentivefp':
        return create_attentivefp_model(
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            global_feat_dim=global_feat_dim,
            device=device,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
            num_timesteps=num_timesteps,
        )
    if model_name == 'directed_mpnn':
        return create_directed_mpnn_model(
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            global_feat_dim=global_feat_dim,
            device=device,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
        )
    raise ValueError(f'Unknown model_name: {model_name}')
