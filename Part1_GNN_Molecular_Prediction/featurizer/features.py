import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges


def one_hot_encoding(value, choices):
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def compute_gasteiger_charges(mol):
    try:
        ComputeGasteigerCharges(mol)
        charges = []
        for atom in mol.GetAtoms():
            if atom.HasProp('_GasteigerCharge'):
                charge = safe_float(atom.GetDoubleProp('_GasteigerCharge'))
            else:
                charge = 0.0
            if np.isnan(charge) or np.isinf(charge):
                charge = 0.0
            charges.append(charge)
        return charges
    except Exception:
        return [0.0] * mol.GetNumAtoms()


def get_global_descriptors(mol):
    try:
        return [
            safe_float(Descriptors.MolLogP(mol)),
            safe_float(Descriptors.MolMR(mol)),
            safe_float(rdMolDescriptors.CalcTPSA(mol)),
            safe_float(rdMolDescriptors.CalcNumHBD(mol)),
            safe_float(rdMolDescriptors.CalcNumHBA(mol)),
            safe_float(Descriptors.NumRotatableBonds(mol)),
            safe_float(Descriptors.RingCount(mol)),
            safe_float(Descriptors.MolWt(mol) / 100.0),
        ]
    except Exception:
        return [0.0] * 8


def featurize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None, None

    mol_h = Chem.AddHs(mol)
    gasteiger_charges = compute_gasteiger_charges(mol_h)
    mol_no_h = Chem.RemoveHs(mol_h)

    node_features = []
    for idx, atom in enumerate(mol_h.GetAtoms()):
        features = []
        features += one_hot_encoding(atom.GetSymbol(), [
            'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
            'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb',
            'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu',
            'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb'
        ])
        features += one_hot_encoding(str(atom.GetChiralTag()), [
            'CHI_UNSPECIFIED', 'CHI_TETRAHEDRAL_CW', 'CHI_TETRAHEDRAL_CCW', 'CHI_OTHER'
        ])
        features += one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        features += one_hot_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8])
        features += one_hot_encoding(atom.GetTotalValence(), [0, 1, 2, 3, 4, 5, 6])
        features += one_hot_encoding(atom.GetFormalCharge(), [-2, -1, 0, 1, 2])
        features.append(gasteiger_charges[idx])
        features += one_hot_encoding(str(atom.GetHybridization()), [
            'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2'
        ])
        features.append(1 if atom.GetIsAromatic() else 0)
        features.append(1 if atom.IsInRing() else 0)

        ring_size_value = 0
        if atom.IsInRingSize(5):
            ring_size_value = 1
        elif atom.IsInRingSize(6):
            ring_size_value = 2
        features += one_hot_encoding(ring_size_value, [0, 1, 2])

        features.append(atom.GetMass() / 100.0)
        features.append(atom.GetAtomicNum() / 100.0)
        node_features.append(features)

    x = torch.tensor(node_features, dtype=torch.float)

    edge_indices, edge_attrs = [], []
    for bond in mol_h.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        b_features = []
        b_features += one_hot_encoding(str(bond.GetBondType()), ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'])
        b_features += one_hot_encoding(str(bond.GetStereo()), [
            'STEREONONE', 'STEREOANY', 'STEREOZ', 'STEREOE', 'STEREOCIS', 'STEREOTRANS'
        ])
        b_features.append(1 if bond.GetIsConjugated() else 0)
        b_features.append(1 if bond.IsInRing() else 0)
        b_features.extend([1 if bond.IsInRingSize(size) else 0 for size in [3, 4, 5, 6, 7, 8]])
        edge_indices += [[i, j], [j, i]]
        edge_attrs += [b_features, b_features]

    if len(edge_indices) > 0:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 20), dtype=torch.float)

    global_feats = torch.tensor(get_global_descriptors(mol_no_h), dtype=torch.float)
    return x, edge_index, edge_attr, global_feats