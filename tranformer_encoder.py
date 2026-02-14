import torch.nn as nn
import torch.nn.functional as F
import torch
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, max_len=128):
        super().__init__()

        self.tok = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.pos = nn.Embedding(max_len, emb_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=8,
            batch_first=True,
            dropout=0.6
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)

        x = self.tok(idx) + self.pos(pos)
        return self.encoder(x)
