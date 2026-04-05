import torch
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from step3_mpnn_model import BasicMPNN, device # আগের ফাইল থেকে মডেল কল করছি

# ১. ডেটাসেট এবং Test Data লোড করা
print("[*] Loading Data for Evaluation...")
dataset = MoleculeNet(root='./data/MoleculeNet', name='ESOL')
torch.manual_seed(12345)
dataset = dataset.shuffle()
test_dataset = dataset[900:] # এই ২২৮টি মলিকিউল মডেল আগে কখনো দেখেনি
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ২. ট্রেইন করা মডেলটি প্রস্তুত করা
print("[*] Initializing Model...")
model = BasicMPNN(dataset.num_node_features).to(device)
criterion = torch.nn.MSELoss()

# (বাস্তব প্রজেক্টে আমরা step3 থেকে মডেলের ওয়েট (weights) লোড করি, 
# তবে এখানে টেস্টিংয়ের জন্য আমরা ইভ্যালুয়েশন লুপটি তৈরি করছি)

# ৩. Evaluation (টেস্টিং) ফাংশন
def test():
    model.eval() # মডেলকে 'টেস্টিং মোডে' রাখা
    total_loss = 0
    correct_predictions = 0
    
    with torch.no_grad(): # টেস্টিংয়ের সময় মডেল নতুন করে শিখবে না
        for data in test_loader:
            data = data.to(device)
            data.x = data.x.float()
            data.y = data.y.float().view(-1, 1)
            
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            total_loss += loss.item()
            
    return total_loss / len(test_loader)

# ৪. ফলাফল দেখা
print("\n[*] Running Evaluation on Test Data...")
test_error = test()
print(f"[*] Mean Squared Error (MSE) on Unknown Data: {test_error:.4f}")
print("[*] Step 4 Complete: Model Evaluation Pipeline is ready!")