import argparse
import json
import os
import sys

import optuna
import torch
import torch.nn as nn
from optuna.samplers import TPESampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset.data_loader import get_dataloaders
from models.model_factory import create_model


def parse_args():
    parser = argparse.ArgumentParser(description='Train molecular property models with Optuna tuning.')
    parser.add_argument('--debug', action='store_true', help='Run a quick sanity check without full training.')
    return parser.parse_args()


def get_data(batch_size=64):
    return get_dataloaders(batch_size=batch_size)


def build_model(config, num_node, num_edge, num_global, device):
    return create_model(
        config['model_name'],
        num_node,
        num_edge,
        num_global,
        device,
        hidden_channels=config['hidden_channels'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        num_timesteps=config.get('num_timesteps', 2),
    )


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        global_feats = data.global_feats if hasattr(data, 'global_feats') else None
        out = model(data.x, data.edge_index, data.edge_attr, data.batch, global_feats)
        loss = criterion(out, data.y.view(-1, 1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * out.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            global_feats = data.global_feats if hasattr(data, 'global_feats') else None
            out = model(data.x, data.edge_index, data.edge_attr, data.batch, global_feats)
            total_loss += criterion(out, data.y.view(-1, 1)).item() * out.size(0)
    return (total_loss / len(loader.dataset)) ** 0.5


def objective(trial, train_loader, valid_loader, num_node, num_edge, num_global, device):
    model_name = trial.suggest_categorical('model_name', ['attentivefp', 'directed_mpnn'])
    hidden_channels = trial.suggest_categorical('hidden_channels', [64, 128, 200, 256])
    num_layers = trial.suggest_int('num_layers', 2, 5)
    dropout = trial.suggest_float('dropout', 0.1, 0.4)
    lr = trial.suggest_float('lr', 1e-4, 5e-3, log=True)

    config = {
        'model_name': model_name,
        'hidden_channels': hidden_channels,
        'num_layers': num_layers,
        'dropout': dropout,
        'lr': lr,
        'num_timesteps': 2,
    }

    model = build_model(config, num_node, num_edge, num_global, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=25, T_mult=2, eta_min=1e-6)
    criterion = nn.MSELoss()

    best_val_rmse = float('inf')
    patience_counter = 0

    for epoch in range(1, 11):
        train_one_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        val_rmse = evaluate(model, valid_loader, criterion, device)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 3:
                break

    return best_val_rmse


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[*] Starting Optuna + Ensemble Execution on Device: {device}")

    if args.debug:
        max_epochs = 3
        patience = 3
        num_models = 1
        batch_size = 16
        num_trials = 2
    else:
        max_epochs = 700
        patience = 60
        num_models = 5
        batch_size = 64
        num_trials = 50

    train_loader, valid_loader, test_loader, num_node, num_edge, num_global = get_data(batch_size=batch_size)

    print(f"\n[*] Starting Optuna Hyperparameter Tuning ({num_trials} Trials)...")
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(n_startup_trials=25, n_ei_candidates=24),
        pruner=optuna.pruners.HyperbandPruner(min_resource=1, reduction_factor=3),
    )
    study.optimize(lambda trial: objective(trial, train_loader, valid_loader, num_node, num_edge, num_global, device), n_trials=num_trials)

    best_params = study.best_params
    print(f"\n[+] Optuna Found Best Hyperparameters: {best_params}")

    best_params_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_params.json')
    with open(best_params_path, 'w') as f:
        json.dump(best_params, f, indent=2)

    ensemble_models = []
    for m in range(num_models):
        print(f"\n==================================================")
        print(f"[*] Training Ensemble Model {m+1}/{num_models} with Best Params")
        print(f"==================================================")

        model = build_model(best_params, num_node, num_edge, num_global, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=best_params['lr'], weight_decay=1e-6)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-6)
        criterion = nn.MSELoss()

        best_val_rmse = float('inf')
        patience_counter = 0
        model_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'sota_model_fold_{m+1}.pth')

        for epoch in range(1, max_epochs + 1):
            train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_rmse = evaluate(model, valid_loader, criterion, device)
            scheduler.step()

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                patience_counter = 0
                torch.save(model.state_dict(), model_save_path)
            else:
                patience_counter += 1

            if epoch % 20 == 0 or epoch == 1:
                print(f'Epoch: {epoch:03d} | Validation RMSE: {val_rmse:.4f}')

            if patience_counter >= patience:
                print(f"[!] Early Stopping triggered at Epoch {epoch}!")
                break

        print(f"[*] Best Validation RMSE for Model {m+1}: {best_val_rmse:.4f}")
        ensemble_models.append(model_save_path)

    print("[*] Optuna + Ensemble Training Complete! Ready for Evaluation.")


if __name__ == '__main__':
    main()
