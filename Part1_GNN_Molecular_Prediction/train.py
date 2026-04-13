import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import optuna
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset.data_loader import get_dataloaders
from models.attentivefp_model import create_attentivefp_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n[*] Starting SOTA Optuna + Ensemble Execution on Device: {device}")

MAX_EPOCHS = 700
PATIENCE = 40
NUM_MODELS = 5
BATCH_SIZE = 64

# Load Data
train_loader, valid_loader, test_loader, num_node, num_edge = get_dataloaders(batch_size=BATCH_SIZE)

# ==========================================
# ১. Optuna Objective Function (Auto-Tuning)
# ==========================================
def objective(trial):
    # Optuna নিজে নিজে এই অপশনগুলো থেকে সেরাটি বেছে নেবে
    hidden_channels = trial.suggest_categorical('hidden_channels', [64, 128, 200, 256])
    num_layers = trial.suggest_int('num_layers', 2, 5)
    dropout = trial.suggest_float('dropout', 0.1, 0.4)
    lr = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
    
    model = create_attentivefp_model(num_node, num_edge, device, hidden_channels=hidden_channels, num_layers=num_layers, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    criterion = nn.MSELoss()
    
    # টিউনিংয়ের সময় সময় বাঁচাতে আমরা মাত্র 50 ইপোক পর্যন্ত চেক করব
    best_val_rmse = float('inf')
    for epoch in range(1, 51):
        model.train()
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = criterion(out, data.y.view(-1, 1))
            loss.backward()
            optimizer.step()
            
        # Validation
        model.eval()
        total_mse = 0
        with torch.no_grad():
            for data in valid_loader:
                data = data.to(device)
                out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                total_mse += criterion(out, data.y.view(-1, 1)).item()
        val_rmse = (total_mse / len(valid_loader)) ** 0.5
        
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            
    return best_val_rmse

print("\n[*] Starting Optuna Hyperparameter Tuning (10 Trials)...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10) # 10 বার ভিন্ন ভিন্ন প্যারামিটার দিয়ে টেস্ট করবে

best_params = study.best_params
print(f"\n[+] Optuna Found Best Hyperparameters: {best_params}")

# ==========================================
# ২. SOTA Ensemble Training Loop (Using Best Params)
# ==========================================
ensemble_models = []

for m in range(NUM_MODELS):
    print(f"\n==================================================")
    print(f"[*] Training Ensemble Model {m+1}/{NUM_MODELS} with Best Params")
    print(f"==================================================")
    
    model = create_attentivefp_model(
        num_node, num_edge, device, 
        hidden_channels=best_params['hidden_channels'], 
        num_layers=best_params['num_layers'], 
        dropout=best_params['dropout']
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'], weight_decay=1e-6)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-5)
    criterion = nn.MSELoss()
    
    best_val_rmse = float('inf')
    patience_counter = 0
    model_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'sota_model_fold_{m+1}.pth')

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = criterion(out, data.y.view(-1, 1))
            loss.backward()
            optimizer.step()
            
        # Validation evaluation
        model.eval()
        total_mse = 0
        with torch.no_grad():
            for data in valid_loader:
                data = data.to(device)
                out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                total_mse += criterion(out, data.y.view(-1, 1)).item()
        val_rmse = (total_mse / len(valid_loader)) ** 0.5
        
        scheduler.step(val_rmse)
        
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1
            
        if epoch % 20 == 0 or epoch == 1:
            print(f'Epoch: {epoch:03d} | Validation RMSE: {val_rmse:.4f}')
            
        if patience_counter >= PATIENCE:
            print(f"[!] Early Stopping triggered at Epoch {epoch}!")
            break
            
    print(f"[*] Best Validation RMSE for Model {m+1}: {best_val_rmse:.4f}")
    ensemble_models.append(model_save_path)

print("[*] Optuna + Ensemble Training Complete! Ready for Evaluation.")