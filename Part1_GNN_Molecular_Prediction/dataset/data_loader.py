import torch
from torch_geometric.datasets import MoleculeNet
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
import os
import sys

# আপনার বানানো featurizer ফাইলটিকে ইম্পোর্ট করার জন্য পথ (Path) ঠিক করে দিচ্ছি
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from featurizer.features import featurize_smiles

def generate_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return scaffold

def scaffold_split(dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1):
    scaffolds = defaultdict(list)
    for idx, data in enumerate(dataset):
        scaffold = generate_scaffold(data.smiles)
        scaffolds[scaffold].append(idx)
    
    # স্ক্যাফোল্ড সাইজ অনুযায়ী সাজানো (সবচেয়ে বড়গুলো আগে)
    scaffold_sets = [scaffolds[scaffold] for scaffold in sorted(scaffolds.keys(), key=lambda k: len(scaffolds[k]), reverse=True)]
    
    train_idx, valid_idx, test_idx = [], [], []
    train_cutoff = int(frac_train * len(dataset))
    valid_cutoff = int((frac_train + frac_valid) * len(dataset))
    
    for scaffold_set in scaffold_sets:
        if len(train_idx) + len(scaffold_set) <= train_cutoff:
            train_idx.extend(scaffold_set)
        elif len(valid_idx) + len(scaffold_set) <= (valid_cutoff - train_cutoff):
            valid_idx.extend(scaffold_set)
        else:
            test_idx.extend(scaffold_set)
            
    return torch.utils.data.Subset(dataset, train_idx), torch.utils.data.Subset(dataset, valid_idx), torch.utils.data.Subset(dataset, test_idx)

def get_dataloaders(batch_size=64):
    print("[*] Loading ESOL Dataset and applying RDKit advanced featurization...")
    # ডেটাসেট সেভ করার লোকেশন
    root_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'MoleculeNet')
    raw_dataset = MoleculeNet(root=root_dir, name='ESOL')

    advanced_dataset = []
    for data in raw_dataset:
        x, edge_index, edge_attr = featurize_smiles(data.smiles)
        if x is not None:
            new_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=data.y, smiles=data.smiles)
            advanced_dataset.append(new_data)
            
    train_set, valid_set, test_set = scaffold_split(advanced_dataset)
    print(f"[*] Professional Scaffold Split Completed!")
    print(f"    Train: {len(train_set)} | Valid: {len(valid_set)} | Test: {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader, advanced_dataset[0].num_node_features, advanced_dataset[0].num_edge_features

# টেস্ট করার জন্য
if __name__ == "__main__":
    train_loader, valid_loader, test_loader, num_node, num_edge = get_dataloaders()
    print(f"\nNode features passing to model: {num_node}")
    print("Dataset Module is working perfectly without errors!")