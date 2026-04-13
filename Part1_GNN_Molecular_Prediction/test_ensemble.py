import torch
import torch.nn as nn
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset.data_loader import get_dataloaders
from models.attentivefp_model import create_attentivefp_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[*] Starting SOTA Evaluation on Device: {device}")

# Load Data
_, _, test_loader, num_node, num_edge = get_dataloaders(batch_size=64)

# ==========================================
# Optuna থেকে পাওয়া সেরা প্যারামিটার (Hardcoded)
# ==========================================
BEST_HIDDEN = 200
BEST_LAYERS = 2
BEST_DROPOUT = 0.23390250105874502

ensemble_models = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), f'sota_model_fold_{i}.pth')
    for i in range(1, 6)
]

def evaluate_models(loaders):
    individual_rmses = []
    
    all_targets = []
    all_preds = torch.zeros((len(loaders.dataset), 1)).to(device)
    
    for i, m_path in enumerate(ensemble_models):
        print(f"[*] Evaluating Model {i+1}/5...")
        
        # সঠিক প্যারামিটার দিয়ে মডেল তৈরি
        model = create_attentivefp_model(
            num_node, num_edge, device, 
            hidden_channels=BEST_HIDDEN, 
            num_layers=BEST_LAYERS, 
            dropout=BEST_DROPOUT
        )
        model.load_state_dict(torch.load(m_path, weights_only=True))
        model.eval()
        
        model_preds = []
        model_targets = []
        
        idx = 0
        with torch.no_grad():
            for data in loaders:
                data = data.to(device)
                out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                
                model_preds.append(out)
                model_targets.append(data.y.view(-1, 1))
                
                batch_size = out.size(0)
                all_preds[idx:idx+batch_size] += out
                
                if i == 0:
                    all_targets.append(data.y.view(-1, 1))
                idx += batch_size
                
        # Calculate RMSE for this single model
        m_preds = torch.cat(model_preds, dim=0)
        m_targets = torch.cat(model_targets, dim=0)
        m_rmse = nn.MSELoss()(m_preds, m_targets).item() ** 0.5
        individual_rmses.append(m_rmse)
        print(f"    -> Model {i+1} RMSE: {m_rmse:.4f}")

    # Calculate Scientific Format (Mean ± Std Dev)
    mean_rmse = np.mean(individual_rmses)
    std_rmse = np.std(individual_rmses)
    
    # Calculate True Ensemble Prediction (Averaging the predictions)
    all_preds = all_preds / len(ensemble_models)
    all_targets = torch.cat(all_targets, dim=0)
    ensemble_rmse = nn.MSELoss()(all_preds, all_targets).item() ** 0.5
    
    return mean_rmse, std_rmse, ensemble_rmse

mean_r, std_r, ens_r = evaluate_models(test_loader)

print(f"\n==========================================================")
print(f"  FINAL SCIENTIFIC RESULT (For Paper Publication)  ")
print(f"  Individual Models Performance: {mean_r:.4f} ± {std_r:.4f}")
print(f"  Ensemble (Averaged Prediction) Test RMSE: {ens_r:.4f}")
print(f"==========================================================")