import torch
from torch_geometric.data import Data

# Check if NVIDIA GPU is available (Since you have RTX 4060)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Simple Message Passing Graph (Nodes and Edges)
# As per Professor's PyG learning guideline
edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)
data = data.to(device)

print("\nGraph Data Structure:")
print(data)