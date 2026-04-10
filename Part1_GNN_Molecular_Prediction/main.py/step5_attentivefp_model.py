import torch
import torch.nn.functional as F
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import AttentiveFP

# ১. Device সেটআপ
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[*] Device: {device}")

# ২. ডেটাসেট লোড করা (ESOL)
dataset = MoleculeNet(root='./data/MoleculeNet', name='ESOL')
print(f"[*] Dataset: {dataset.name} | Total Molecules: {len(dataset)}")

# (গাইডলাইন অনুযায়ী Scaffold Split করা যায়, তবে Freshman Assignment-এ আপাতত Random Split দিয়ে মডেল রিপ্রোডিউস করছি)
torch.manual_seed(12345)
dataset = dataset.shuffle()
train_dataset = dataset[:900]
test_dataset = dataset[900:]

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ৩. প্রফেসরের নির্দেশিত उन्नत AttentiveFP মডেল তৈরি
# এটি সাধারণ GCN এর চেয়ে শক্তিশালী, কারণ এটি Edge Feature (বন্ডের তথ্য) ও Attention মেকানিজম ব্যবহার করে
model = AttentiveFP(
    in_channels=dataset.num_node_features,   # পরমাণুর ফিচার (যেমন: C, H, O)
    hidden_channels=64,
    out_channels=1,                          # প্রেডিকশন আউটপুট (Solubility)
    edge_dim=dataset.num_edge_features,      # বন্ডের ফিচার (যেমন: Single/Double bond)
    num_layers=2,                            # Message passing layers
    num_timesteps=2                          # Readout timesteps
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = torch.nn.MSELoss()

# ৪. Training ফাংশন
def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        
        # ডেটা টাইপ Long থেকে Float-এ কনভার্ট (যেমনটা আগের এররে শিখেছিলাম)
        data.x = data.x.float()
        data.edge_attr = data.edge_attr.float() # AttentiveFP Edge Feature ব্যবহার করে
        data.y = data.y.float().view(-1, 1)
        
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch) # Edge attr যোগ করা হলো
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# ৫. Evaluation ফাংশন (আপনার গাইডলাইন অনুযায়ী RMSE বের করার জন্য)
def test(loader):
    model.eval()
    total_mse = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            data.x = data.x.float()
            data.edge_attr = data.edge_attr.float()
            data.y = data.y.float().view(-1, 1)
            
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            total_mse += criterion(out, data.y).item()
            
    mse = total_mse / len(loader)
    rmse = mse ** 0.5 # Root Mean Square Error (RMSE)
    return rmse

# ৬. মডেল ট্রেনিং এবং টেস্টিং রান করা
print("\n[*] Starting Training for AttentiveFP Model (20 Epochs)...")
for epoch in range(1, 21):
    loss = train()
    if epoch % 2 == 0:
        test_rmse = test(test_loader)
        print(f'Epoch: {epoch:03d}, Train Loss (MSE): {loss:.4f}, Test RMSE: {test_rmse:.4f}')

print("\n[*] AttentiveFP Model successfully implemented and evaluated!")