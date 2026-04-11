import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset.data_loader import get_dataloaders
from models.attentivefp_model import create_attentivefp_model

# ==========================================
# ১. SOTA Hyperparameters Setup
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n[*] Starting SOTA Ensemble Execution on Device: {device}")

MAX_EPOCHS = 700
PATIENCE = 40        # Increased patience since LR will reduce before stopping
NUM_MODELS = 5       # Ensemble 5 models
BATCH_SIZE = 64

# Best Tuned Hyperparameters
HYPERPARAMS = {
    'hidden_channels': 200,
    'num_layers': 4,
    'dropout': 0.1,  # Lower dropout for better SOTA fitting
    'lr': 0.003,     # Starting LR (Scheduler will reduce it dynamically)
    'weight_decay': 1e-6
}

print(f"[*] Hyperparameters configuration: {HYPERPARAMS}")

# Load Data
train_loader, valid_loader, test_loader, num_node, num_edge = get_dataloaders(batch_size=BATCH_SIZE)
print(f"[*] Upgraded Node Features: {num_node} | Edge Features: {num_edge}")

# ==========================================
# ২. Core Evaluation Function
# ==========================================
def evaluate_epoch(model, loader):
    model.eval()
    total_mse = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            total_mse += nn.MSELoss()(out, data.y.view(-1, 1)).item()
    return (total_mse / len(loader)) ** 0.5

# ==========================================
# ৩. SOTA Ensemble Training Loop
# ==========================================
ensemble_models = []

for m in range(NUM_MODELS):
    print(f"\n==================================================")
    print(f"[*] Training Ensemble Model {m+1}/{NUM_MODELS}")
    print(f"==================================================")
    
    # Initialize fresh model and optimizer for each iteration
    model = create_attentivefp_model(
        num_node, num_edge, device, 
        hidden_channels=HYPERPARAMS['hidden_channels'], 
        num_layers=HYPERPARAMS['num_layers'], 
        dropout=HYPERPARAMS['dropout']
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=HYPERPARAMS['lr'], weight_decay=HYPERPARAMS['weight_decay'])
    
    # SOTA: ReduceLROnPlateau Scheduler
    # যদি ১০ ইপোক ধরে লস না কমে, তাহলে Learning Rate অর্ধেক (0.5) করে দেবে!
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-5, verbose=True)
    criterion = nn.MSELoss()
    
    best_val_rmse = float('inf')
    patience_counter = 0
    model_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'sota_model_fold_{m+1}.pth')

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_train_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = criterion(out, data.y.view(-1, 1))
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            
        train_mse = total_train_loss / len(train_loader)
        val_rmse = evaluate_epoch(model, valid_loader)
        
        # Step the scheduler (এটি চেক করবে লস কমছে কি না)
        scheduler.step(val_rmse)
        
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1
            
        if epoch % 10 == 0 or epoch == 1:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch: {epoch:03d} | LR: {current_lr:.6f} | Train MSE: {train_mse:.4f} | Validation RMSE: {val_rmse:.4f}')
            
        if patience_counter >= PATIENCE:
            print(f"[!] Early Stopping triggered at Epoch {epoch}! No improvement for {PATIENCE} epochs.")
            break
            
    print(f"[*] Best Validation RMSE for Model {m+1}: {best_val_rmse:.4f}")
    ensemble_models.append(model_save_path)

# ==========================================
# ৪. Final Ensemble Testing
# ==========================================
print("\n[*] Evaluating ENSEMBLE Predictions on Unseen Test Data...")

def ensemble_predict(loaders):
    all_targets = []
    all_preds = torch.zeros((len(loaders.dataset), 1)).to(device)
    
    # ৫টি মডেলের প্রেডিকশন বের করে যোগ করা হচ্ছে
    for m_path in ensemble_models:
        model = create_attentivefp_model(num_node, num_edge, device, hidden_channels=HYPERPARAMS['hidden_channels'], num_layers=HYPERPARAMS['num_layers'], dropout=HYPERPARAMS['dropout'])
        model.load_state_dict(torch.load(m_path, weights_only=True)) # weights_only=True fixed warning
        model.eval()
        
        idx = 0
        with torch.no_grad():
            for data in loaders:
                data = data.to(device)
                out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                
                batch_size = out.size(0)
                all_preds[idx:idx+batch_size] += out
                
                if len(ensemble_models) == 1: # Just to collect targets once
                    all_targets.append(data.y.view(-1, 1))
                idx += batch_size
                
    # সব প্রেডিকশনের গড় (Average) বের করা (This is the magic of Ensembling!)
    all_preds = all_preds / NUM_MODELS
    all_targets = torch.cat(all_targets, dim=0)
    
    final_mse = nn.MSELoss()(all_preds, all_targets).item()
    return final_mse ** 0.5

final_test_rmse = ensemble_predict(test_loader)

print(f"\n==========================================================")
print(f"  FINAL SOTA RESULT: 5-Model Ensemble Evaluated  ")
print(f"  Ensemble Test RMSE on Scaffold Split: {final_test_rmse:.4f}")
print(f"==========================================================")
print("[*] Project Upgraded to True SOTA Standards!")