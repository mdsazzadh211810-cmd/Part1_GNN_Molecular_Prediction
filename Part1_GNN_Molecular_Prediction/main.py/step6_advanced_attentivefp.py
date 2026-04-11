import torch
import torch.nn.functional as F
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import AttentiveFP
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict

# ==========================================
# ১. Device এবং Hyperparameters
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[*] Device: {device}")
MAX_EPOCHS = 500
PATIENCE = 30 

# ==========================================
# ২. RDKit Feature Engineering (Fixed Deprecation & UnboundLocalError)
# ==========================================
def one_hot_encoding(value, choices):
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding

def featurize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node Features (Atom level)
    node_features = []
    for atom in mol.GetAtoms():
        features = []
        features += one_hot_encoding(atom.GetSymbol(), ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb'])
        features += one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        features += one_hot_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8])
        # FIX: Changed GetImplicitValence() to GetValence(getExplicit=False) as per warning
        features += one_hot_encoding(atom.GetValence(getExplicit=False), [0, 1, 2, 3, 4, 5, 6])
        features.append(1 if atom.GetIsAromatic() else 0)
        features.append(atom.GetMass() / 100.0) 
        node_features.append(features)
    x = torch.tensor(node_features, dtype=torch.float)

    # Edge Features (Bond level)
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
        # FIX: If molecule has no bonds, set edge_attr to empty tensor with correct feature dimension (7)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 7), dtype=torch.float)

    return x, edge_index, edge_attr

# ==========================================
# ৩. Scaffold Split Algorithm
# ==========================================
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

# ==========================================
# ৪. Data Processing & Splitting
# ==========================================
print("[*] Downloading and Re-featurizing ESOL with Advanced RDKit Features...")
raw_dataset = MoleculeNet(root='./data/MoleculeNet', name='ESOL')

advanced_dataset = []
for data in raw_dataset:
    x, edge_index, edge_attr = featurize_smiles(data.smiles)
    if x is not None:
        new_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=data.y, smiles=data.smiles)
        advanced_dataset.append(new_data)

print(f"[*] New Node Features: {advanced_dataset[0].num_node_features}")
print(f"[*] New Edge Features: {advanced_dataset[0].num_edge_features}")

train_set, valid_set, test_set = scaffold_split(advanced_dataset)
print(f"[*] Splitting Data (Scaffold Split) -> Train: {len(train_set)}, Valid: {len(valid_set)}, Test: {len(test_set)}")

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=64, shuffle=False)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

# ==========================================
# ৫. Model, Optimizer & Early Stopping
# ==========================================
model = AttentiveFP(
    in_channels=advanced_dataset[0].num_node_features,
    hidden_channels=128, 
    out_channels=1,
    edge_dim=advanced_dataset[0].num_edge_features,
    num_layers=3,        
    num_timesteps=2,
    dropout=0.2          
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-5)
criterion = torch.nn.MSELoss()

def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = criterion(out, data.y.view(-1, 1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate(loader):
    model.eval()
    total_mse = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            total_mse += criterion(out, data.y.view(-1, 1)).item()
    return (total_mse / len(loader)) ** 0.5 

# ==========================================
# ৬. Training Loop with Early Stopping
# ==========================================
print(f"\n[*] Starting Training for max {MAX_EPOCHS} Epochs with Early Stopping...")
best_val_rmse = float('inf')
patience_counter = 0

for epoch in range(1, MAX_EPOCHS + 1):
    train_loss = train()
    val_rmse = evaluate(valid_loader)
    
    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        patience_counter = 0
        torch.save(model.state_dict(), 'best_attentivefp_model.pth')
    else:
        patience_counter += 1
        
    if epoch % 5 == 0 or epoch == 1:
        print(f'Epoch: {epoch:03d}, Train MSE: {train_loss:.4f}, Val RMSE: {val_rmse:.4f}')
        
    if patience_counter >= PATIENCE:
        print(f"\n[!] Early Stopping triggered at Epoch {epoch}! No improvement for {PATIENCE} epochs.")
        break

# Load the best model to test
model.load_state_dict(torch.load('best_attentivefp_model.pth'))
test_rmse = evaluate(test_loader)
print(f"\n[*] Final Test RMSE on Scaffold Split (Unseen Data): {test_rmse:.4f}")
print("[*] Advanced Project Step Complete!")