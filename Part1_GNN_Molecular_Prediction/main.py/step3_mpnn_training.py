import torch
from torch_geometric.datasets import MoleculeNet

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[*] Running on device: {device}\n")

# ডাউনলোড করার ফোল্ডার সেট করা (আপনার E ড্রাইভে ডেটা সেভ হবে)
data_path = './data/MoleculeNet'

# PyG-এর ওপেন-সোর্স MoleculeNet থেকে 'ESOL' (Solubility Prediction) ডেটাসেট ডাউনলোড করা
print("[*] Downloading open-source MoleculeNet (ESOL) dataset as per PyG standard...")
dataset = MoleculeNet(root=data_path, name='ESOL')

print("\n[*] Dataset Overview:")
print(f"Dataset Name: {dataset.name}")
print(f"Number of Graphs (Molecules): {len(dataset)}")
print(f"Number of Node Features: {dataset.num_node_features}")
print(f"Number of Edge Features: {dataset.num_edge_features}")

# প্রথম মলিকিউলটির (Graph) ডেটা চেক করা
data = dataset[0]
print("\n[*] First Molecule Graph Data:")
print(data)

print("\n[*] Dataset successfully loaded and ready for MPNN Model!")