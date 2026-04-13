import torch
import torch.nn as nn


class VirtualNodeAttentionReadout(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.virtual_node = nn.Parameter(torch.randn(1, hidden_dim))
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, batch):
        batch_size = batch.max().item() + 1
        virtual_nodes = self.virtual_node.expand(batch_size, -1)
        outputs = []

        for i in range(batch_size):
            mask = batch == i
            graph_x = x[mask].unsqueeze(0)
            virtual = virtual_nodes[i].unsqueeze(0).unsqueeze(0)
            if graph_x.size(1) == 0:
                outputs.append(virtual.squeeze(0))
                continue
            attn_out, _ = self.attention(virtual, graph_x, graph_x)
            attended = attn_out.squeeze(0)
            attended = self.norm1(attended)
            out = self.ffn(attended)
            out = self.norm2(out + attended)
            outputs.append(out.squeeze(0))

        return torch.stack(outputs, dim=0)
