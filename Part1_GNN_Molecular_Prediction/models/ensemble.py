import torch
import torch.nn as nn


class HeterogeneousEnsemble(nn.Module):
    def __init__(self, model_dict):
        super().__init__()
        self.models = nn.ModuleDict(model_dict)
        self.ensemble_weights = nn.Parameter(torch.ones(len(model_dict)))

    def forward(self, x, edge_index, edge_attr, batch, global_feats=None):
        predictions = []
        for model in self.models.values():
            predictions.append(model(x, edge_index, edge_attr, batch, global_feats))

        stacked = torch.stack(predictions, dim=0)
        weights = torch.softmax(self.ensemble_weights, dim=0).view(-1, 1, 1)
        return (stacked * weights).sum(dim=0)

    def load_state_dicts(self, state_dict_map):
        for name, path in state_dict_map.items():
            if name in self.models and path is not None:
                self.models[name].load_state_dict(torch.load(path))
