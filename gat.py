import torch.nn as nn
import torch.nn.functional as F
import torch
class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.2, alpha=0.2):
        super().__init__()

        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(2 * out_dim, 1, bias=False)

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        """
        x: (V, D)
        adj: sparse COO (V, V)
        """
        Wh = self.W(x)  # (V, D)

        row, col = adj.indices()
        Wh_i = Wh[row]
        Wh_j = Wh[col]

        e = self.leakyrelu(
            self.a(torch.cat([Wh_i, Wh_j], dim=1))
        ).squeeze()

        attn = torch.sparse_coo_tensor(
            adj.indices(),
            e,
            adj.size()
        )

        attn = torch.sparse.softmax(attn, dim=1)
        attn = torch.sparse_coo_tensor(
            attn.indices(),
            self.dropout(attn.values()),
            attn.size()
        )

        out = torch.sparse.mm(attn, Wh)
        return F.elu(out)

class VocabGAT(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.gat = GATLayer(emb_dim, emb_dim)

    def forward(self, emb_weight, adj):
        return self.gat(emb_weight, adj)
