import torch
import torch.nn as nn
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset.data_loader import get_dataloaders
from models.attentivefp_model import create_attentivefp_model

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[*] Starting Evaluation on Device: {device}")

# Same Best Tuned Hyperparameters
HYPERPARAMS = {'hidden_channels': 200, 'num_layers': 4, 'dropout': 0.1}

# Load Data (Only need test_loader, but function returns all)
_, _, test_loader, num_node, num_edge = get_dataloaders(batch_size=64)

# সেভ করা ৫টি মডেলের লোকেশন
ensemble_models = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), f'sota_model_fold_{i}.pth')
    for i in range(1, 6)
]

def ensemble_predict(loaders):
    all_targets = []
    all_preds = torch.zeros((len(loaders.dataset), 1)).to(device)
    
    for i, m_path in enumerate(ensemble_models):
        print(f"[*] Loading trained model {i+1}/5 from: {os.path.basename(m_path)}")
        model = create_attentivefp_model(num_node, num_edge, device, hidden_channels=HYPERPARAMS['hidden_channels'], num_layers=HYPERPARAMS['num_layers'], dropout=HYPERPARAMS['dropout'])
        model.load_state_dict(torch.load(m_path, weights_only=True))
        model.eval()
        
        idx = 0
        with torch.no_grad():
            for data in loaders:
                data = data.to(device)
                out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                
                batch_size = out.size(0)
                all_preds[idx:idx+batch_size] += out
                
                # FIX: শুধু প্রথম মডেল রান হওয়ার সময় টার্গেটগুলো একবার সেভ করা হবে
                if i == 0: 
                    all_targets.append(data.y.view(-1, 1))
                idx += batch_size
                
    # ৫টি প্রেডিকশনের গড় (Average)
    all_preds = all_preds / len(ensemble_models)
    all_targets = torch.cat(all_targets, dim=0)
    
    final_mse = nn.MSELoss()(all_preds, all_targets).item()
    return final_mse ** 0.5

# Run final evaluation
final_test_rmse = ensemble_predict(test_loader)

print(f"\n==========================================================")
print(f"  FINAL SOTA RESULT: 5-Model Ensemble Evaluated  ")
print(f"  Ensemble Test RMSE on Scaffold Split: {final_test_rmse:.4f}")
print(f"==========================================================")