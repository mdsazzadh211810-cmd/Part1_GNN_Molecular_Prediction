import torch
from rdkit import Chem

# One-hot encoding ফাংশন
def one_hot_encoding(value, choices):
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding

# মলিকিউল থেকে Node (Atom) এবং Edge (Bond) ফিচার বের করার ফাংশন
def featurize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None

    # === Node Features (Atom level) ===
    node_features = []
    for atom in mol.GetAtoms():
        features = []
        features += one_hot_encoding(atom.GetSymbol(), ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb'])
        features += one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        features += one_hot_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8])
        
        # [FIXED ERROR]: C++ Signature mismatch এরর সলভ করার জন্য GetTotalValence ব্যবহার করা হলো
        features += one_hot_encoding(atom.GetTotalValence(), [0, 1, 2, 3, 4, 5, 6])
        
        features.append(1 if atom.GetIsAromatic() else 0)
        features.append(atom.GetMass() / 100.0) 
        node_features.append(features)
        
    x = torch.tensor(node_features, dtype=torch.float)

    # === Edge Features (Bond level) ===
    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()
        
        b_features = []
        b_features += one_hot_encoding(str(bond_type), ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'])
        b_features.append(1 if bond.GetIsConjugated() else 0)
        b_features.append(1 if bond.IsInRing() else 0)
        
        edge_indices += [[i, j], [j, i]]
        edge_attrs += [b_features, b_features]

    if len(edge_indices) > 0:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    else:
        # মলিকিউলে কোনো বন্ড না থাকলে ডাইমেনশন ঠিক রাখার জন্য (7 bond features)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 7), dtype=torch.float)

    return x, edge_index, edge_attr

# কোডটি ঠিকমতো কাজ করছে কি না তা চেক করার জন্য ছোট্ট টেস্ট
if __name__ == "__main__":
    print("Testing Featurizer with a sample SMILES...")
    sample_smiles = "CCO" # Ethanol
    x, edge_idx, edge_attr = featurize_smiles(sample_smiles)
    print(f"Node Feature Shape: {x.shape}")
    print(f"Edge Index Shape: {edge_idx.shape}")
    print(f"Edge Attribute Shape: {edge_attr.shape}")
    print("Featurizer is working perfectly without errors!")