import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset.data_loader import get_dataloaders
from models.model_factory import create_model


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate ensemble models or run a quick forward-pass sanity check.')
    parser.add_argument('--debug', action='store_true', help='Run a quick forward-pass test instead of loading saved weights.')
    return parser.parse_args()


def load_best_params():
    best_params_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_params.json')
    if os.path.exists(best_params_path):
        with open(best_params_path, 'r') as f:
            best_params = json.load(f)
    else:
        best_params = {
            'model_name': 'attentivefp',
            'hidden_channels': 200,
            'num_layers': 2,
            'dropout': 0.234,
        }
    return best_params


def evaluate_models(loader, device, best_params):
    num_node = loader.dataset[0].num_node_features
    num_edge = loader.dataset[0].num_edge_features
    num_global = loader.dataset[0].global_feats.shape[1]

    ensemble_models = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), f'sota_model_fold_{i}.pth')
        for i in range(1, 6)
    ]

    individual_rmses = []
    all_targets = []
    all_preds = torch.zeros((len(loader.dataset), 1), device=device)

    for i, m_path in enumerate(ensemble_models):
        print(f"[*] Evaluating Model {i+1}/5...")

        model = create_model(
            best_params.get('model_name', 'attentivefp'),
            num_node,
            num_edge,
            num_global,
            device,
            hidden_channels=best_params.get('hidden_channels', 200),
            num_layers=best_params.get('num_layers', 2),
            dropout=best_params.get('dropout', 0.234),
        )

        if not os.path.exists(m_path):
            raise FileNotFoundError(f'Model checkpoint not found: {m_path}')

        try:
            state = torch.load(m_path)
            model.load_state_dict(state)
        except Exception as exc:
            raise RuntimeError(f'Failed to load model state from {m_path}: {exc}')

        model.eval()

        model_preds = []
        model_targets = []
        idx = 0

        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                global_feats = data.global_feats if hasattr(data, 'global_feats') else None
                out = model(data.x, data.edge_index, data.edge_attr, data.batch, global_feats)

                model_preds.append(out)
                model_targets.append(data.y.view(-1, 1))

                batch_size = out.size(0)
                all_preds[idx:idx + batch_size] += out
                if i == 0:
                    all_targets.append(data.y.view(-1, 1))
                idx += batch_size

        m_preds = torch.cat(model_preds, dim=0)
        m_targets = torch.cat(model_targets, dim=0)
        m_rmse = nn.MSELoss()(m_preds, m_targets).item() ** 0.5
        individual_rmses.append(m_rmse)
        print(f"    -> Model {i+1} RMSE: {m_rmse:.4f}")

    mean_rmse = np.mean(individual_rmses)
    std_rmse = np.std(individual_rmses)
    all_preds = all_preds / len(ensemble_models)
    all_targets = torch.cat(all_targets, dim=0)
    ensemble_rmse = nn.MSELoss()(all_preds, all_targets).item() ** 0.5

    return mean_rmse, std_rmse, ensemble_rmse


def run_debug(loader, device):
    data = next(iter(loader))
    num_node = data.num_node_features
    num_edge = data.num_edge_features
    num_global = data.global_feats.shape[1]
    model = create_model('attentivefp', num_node, num_edge, num_global, device, hidden_channels=64, num_layers=2, dropout=0.2)
    model.eval()
    with torch.no_grad():
        out = model(data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device), data.batch.to(device), data.global_feats.to(device))
    print(f"Debug forward pass successful: output shape {out.shape}")


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Starting Evaluation on Device: {device}")
    _, _, test_loader, _, _, _ = get_dataloaders(batch_size=64)

    if args.debug:
        run_debug(test_loader, device)
        return

    best_params = load_best_params()
    mean_r, std_r, ens_r = evaluate_models(test_loader, device, best_params)

    print(f"\n==========================================================")
    print(f"  FINAL EVALUATION")
    print(f"  Individual Models Performance: {mean_r:.4f} ± {std_r:.4f}")
    print(f"  Ensemble (Averaged Prediction) Test RMSE: {ens_r:.4f}")
    print(f"==========================================================")


if __name__ == '__main__':
    main()
