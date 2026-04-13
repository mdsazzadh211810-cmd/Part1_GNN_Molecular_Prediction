import os
import sys
import random
from collections import defaultdict

import torch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.datasets import MoleculeNet
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

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


def scaffold_k_fold_split(dataset, n_splits=5):
    scaffolds = defaultdict(list)
    for idx, data in enumerate(dataset):
        scaffold = generate_scaffold(data.smiles)
        scaffolds[scaffold].append(idx)

    scaffold_sets = sorted(scaffolds.values(), key=len, reverse=True)
    folds = [[] for _ in range(n_splits)]
    for idx, scaffold_set in enumerate(scaffold_sets):
        folds[idx % n_splits].extend(scaffold_set)
    return folds


def get_dataloaders(batch_size=64):
    print("[*] Loading ESOL Dataset and applying RDKit advanced featurization...")
    root_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'MoleculeNet')
    raw_dataset = MoleculeNet(root=root_dir, name='ESOL')

    advanced_dataset = []
    for data in raw_dataset:
        x, edge_index, edge_attr, global_feats = featurize_smiles(data.smiles)
        if x is not None:
            advanced_dataset.append(Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                global_feats=global_feats.unsqueeze(0),
                y=data.y,
                smiles=data.smiles,
            ))

    train_set, valid_set, test_set = scaffold_split(advanced_dataset)
    print(f"[*] Professional Scaffold Split Completed!")
    print(f"    Train: {len(train_set)} | Valid: {len(valid_set)} | Test: {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return (
        train_loader,
        valid_loader,
        test_loader,
        advanced_dataset[0].num_node_features,
        advanced_dataset[0].num_edge_features,
        advanced_dataset[0].global_feats.shape[1],
    )


def get_k_fold_dataloaders(batch_size=64, n_splits=5):
    print(f"[*] Loading ESOL Dataset for {n_splits}-fold scaffold CV...")
    root_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'MoleculeNet')
    raw_dataset = MoleculeNet(root=root_dir, name='ESOL')

    advanced_dataset = []
    for data in raw_dataset:
        x, edge_index, edge_attr, global_feats = featurize_smiles(data.smiles)
        if x is not None:
            advanced_dataset.append(Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                global_feats=global_feats.unsqueeze(0),
                y=data.y,
                smiles=data.smiles,
            ))

    folds = scaffold_k_fold_split(advanced_dataset, n_splits=n_splits)
    fold_loaders = []
    for fold in range(n_splits):
        test_idx = folds[fold]
        train_valid_idx = [idx for i, fold_idxs in enumerate(folds) if i != fold for idx in fold_idxs]
        random.Random(42).shuffle(train_valid_idx)
        split = int(len(train_valid_idx) * 0.9)
        train_idx = train_valid_idx[:split]
        valid_idx = train_valid_idx[split:]

        train_loader = DataLoader(torch.utils.data.Subset(advanced_dataset, train_idx), batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(torch.utils.data.Subset(advanced_dataset, valid_idx), batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(torch.utils.data.Subset(advanced_dataset, test_idx), batch_size=batch_size, shuffle=False)
        fold_loaders.append((train_loader, valid_loader, test_loader))

    return fold_loaders, advanced_dataset[0].num_node_features, advanced_dataset[0].num_edge_features, advanced_dataset[0].global_feats.shape[1]


if __name__ == "__main__":
    train_loader, valid_loader, test_loader, num_node, num_edge, num_global = get_dataloaders()
    print(f"\nNode features passing to model: {num_node}")
    print(f"Edge features passing to model: {num_edge}")
    print(f"Global features passing to model: {num_global}")
    print("Dataset Module is working perfectly without errors!")