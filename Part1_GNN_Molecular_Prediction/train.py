import torch
import torch.nn as nn
import os
import sys

# অন্য ফোল্ডারের মডিউলগুলো ইম্পোর্ট করার জন্য পথ ঠিক করা
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset.data_loader import get_dataloaders
from models.attentivefp_model import create_attentivefp_model

# ==========================================
# ১. Setup and Hyperparameters
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n[*] Starting Project Execution on Device: {device}")

MAX_EPOCHS = 500
PATIENCE = 30
LR = 0.002
WEIGHT_DECAY = 1e-5

# ==========================================
# ২. Load Data & Initialize Model
# ==========================================
train_loader, valid_loader, test_loader, num_node, num_edge = get_dataloaders(batch_size=64)

model = create_attentivefp_model(num_node, num_edge, device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = nn.MSELoss()

# ==========================================
# ৩. Training & Evaluation Functions
# ==========================================
def train_epoch():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        # মডেলকে ফিচারগুলো দেওয়া হচ্ছে
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = criterion(out, data.y.view(-1, 1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate_epoch(loader):
    model.eval()
    total_mse = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            total_mse += criterion(out, data.y.view(-1, 1)).item()
    return (total_mse / len(loader)) ** 0.5  # Returning RMSE

# ==========================================
# ৪. Execution Loop with Early Stopping
# ==========================================
print(f"\n[*] Commencing Training for max {MAX_EPOCHS} Epochs with Early Stopping...")
best_val_rmse = float('inf')
patience_counter = 0
best_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_attentivefp_model.pth')

for epoch in range(1, MAX_EPOCHS + 1):
    train_mse = train_epoch()
    val_rmse = evaluate_epoch(valid_loader)
    
    # Validation Error কমলে মডেল সেভ হবে
    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        patience_counter = 0
        torch.save(model.state_dict(), best_model_path)
    else:
        patience_counter += 1
        
    if epoch % 5 == 0 or epoch == 1:
        print(f'Epoch: {epoch:03d} | Train MSE: {train_mse:.4f} | Validation RMSE: {val_rmse:.4f}')
        
    # যদি মডেল অনেকক্ষণ ধরে উন্নতি না করে, তবে মাঝপথেই থেমে যাবে
    if patience_counter >= PATIENCE:
        print(f"\n[!] Early Stopping triggered at Epoch {epoch}! No improvement for {PATIENCE} epochs.")
        break

# ==========================================
# ৫. Final Testing on Unseen Data
# ==========================================
print("\n[*] Loading best saved model for final evaluation on Test Data...")
model.load_state_dict(torch.load(best_model_path))
test_rmse = evaluate_epoch(test_loader)

print(f"==================================================")
print(f"  FINAL RESULT: SOTA AttentiveFP Model evaluated  ")
print(f"  Test RMSE on Scaffold Split (Unseen Data): {test_rmse:.4f}")
print(f"==================================================")
print("[*] Part 1 of Freshman Assignment Complete! Ready to push to GitHub.")