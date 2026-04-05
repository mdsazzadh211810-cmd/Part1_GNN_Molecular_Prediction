import torch
from torch_geometric.data import Data
from rdkit import Chem

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[*] Running on device: {device}\n")

# প্রফেসরের ইনস্ট্রাকশন অনুযায়ী: RDKit দিয়ে মলিকিউল থেকে গ্রাফ তৈরি (For MPNN)
def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print("Invalid SMILES!")
        return None

    # Node Features (যেমন: পরমাণুর ধরন বা Atomic Number)
    x = []
    for atom in mol.GetAtoms():
        x.append([atom.GetAtomicNum()])
    x = torch.tensor(x, dtype=torch.float)

    # Edge Index (যেমন: পরমাণুগুলোর মধ্যে বন্ড বা সংযোগ)
    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        # যেহেতু গ্রাফটি Undirected, তাই দুই দিকেই বন্ড যোগ করতে হবে
        edge_index += [[i, j], [j, i]] 
    
    if len(edge_index) > 0:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index)

# একটি সাধারণ মলিকিউল (ইথানল - CCO) দিয়ে টেস্ট করা
sample_smiles = "CCO"
print(f"[*] Converting SMILES: '{sample_smiles}' to PyG Graph...")

graph_data = smiles_to_graph(sample_smiles)
graph_data = graph_data.to(device)

print("\n[*] MPNN Model Input (Graph Data):")
print(graph_data)