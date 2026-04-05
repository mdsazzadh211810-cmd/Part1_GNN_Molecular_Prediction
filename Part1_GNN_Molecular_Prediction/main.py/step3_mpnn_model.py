import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# ১. Device এবং ডেটাসেট সেটআপ
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[*] Device: {device}")

dataset = MoleculeNet(root='./data/MoleculeNet', name='ESOL')
print(f"[*] Dataset: {dataset.name} | Total Molecules: {len(dataset)}")

# ডেটাসেটকে দুই ভাগে ভাগ করা
torch.manual_seed(12345)
dataset = dataset.shuffle()
train_dataset = dataset[:900]
test_dataset = dataset[900:]

# DataLoader তৈরি
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ২. MPNN মডেল তৈরি
class BasicMPNN(torch.nn.Module):
    def __init__(self, num_node_features):
        super(BasicMPNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 64)
        self.conv3 = GCNConv(64, 64)
        self.lin = Linear(64, 1)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x

model = BasicMPNN(dataset.num_node_features).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

# ৩. Training ফাংশন (Here is the FIX for the RuntimeError)
def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        
        # [FIX] ডেটা টাইপ Long থেকে Float এ কনভার্ট করা হচ্ছে
        data.x = data.x.float() 
        data.y = data.y.float().view(-1, 1) # y কেও ফ্লোট এবং সঠিক শেপে আনা হলো
        
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# ৪. মডেল ট্রেনিং রান করা
print("\n[*] Starting Training for 20 Epochs...")
for epoch in range(1, 21):
    loss = train()
    if epoch % 2 == 0:
        print(f'Epoch: {epoch:03d}, Loss (Error): {loss:.4f}')

print("\n[*] Training Complete! Your MPNN model successfully learned from the chemical structures.")