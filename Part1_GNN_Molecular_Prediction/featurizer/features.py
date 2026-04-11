import torch
from rdkit import Chem

def one_hot_encoding(value, choices):
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding

def featurize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None

    # === Node Features (Atom level) -> 88 Features ===
    node_features = []
    for atom in mol.GetAtoms():
        features = []
        features += one_hot_encoding(atom.GetSymbol(), ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb'])
        features += one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        features += one_hot_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8])
        features += one_hot_encoding(atom.GetTotalValence(), [0, 1, 2, 3, 4, 5, 6])
        features += one_hot_encoding(atom.GetFormalCharge(), [-1, -2, 1, 2, 0])
        features += one_hot_encoding(str(atom.GetHybridization()), ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2'])
        
        features.append(1 if atom.GetIsAromatic() else 0)
        features.append(atom.GetMass() / 100.0) 
        features.append(1 if atom.IsInRing() else 0)
        node_features.append(features)
        
    x = torch.tensor(node_features, dtype=torch.float)

    # === Edge Features (Bond level) -> 20 Features ===
    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        
        b_features = []
        b_features += one_hot_encoding(str(bond.GetBondType()), ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'])
        b_features += one_hot_encoding(str(bond.GetStereo()), ['STEREONONE', 'STEREOANY', 'STEREOZ', 'STEREOE'])
        
        b_features.append(1 if bond.GetIsConjugated() else 0)
        b_features.append(1 if bond.IsInRing() else 0)
        
        # Ring Sizes (3 to 8)
        b_features.extend([1 if bond.IsInRingSize(s) else 0 for s in [3, 4, 5, 6, 7, 8]])
        
        edge_indices += [[i, j], [j, i]]
        edge_attrs += [b_features, b_features]

    # এখানে ইনডেন্টেশন (স্পেস) ঠিক করা হয়েছে
    if len(edge_indices) > 0:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 18), dtype=torch.float) 

    return x, edge_index, edge_attr