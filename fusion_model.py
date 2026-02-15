import torch.nn as nn
import torch.nn.functional as F
import torch
from tranformer_encoder import TransformerEncoder
from gat import VocabGAT
class HumanAIDetector(nn.Module):
    def __init__(self, vocab_size, emb_dim, adj_sparse):
        super().__init__()
        self.text_enc = TransformerEncoder(vocab_size, emb_dim)
        self.gat_base_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.vocab_gat = VocabGAT(emb_dim)
        self.adj = adj_sparse
        
        # Adaptive Gate
        # This allows the model to decide the importance of each branch
        self.gate = nn.Parameter(torch.tensor([0.5])) 
        
        self.fusion_layer = nn.Linear(emb_dim * 2, emb_dim)
        self.classifier = nn.Linear(emb_dim, 2)
        self.register_buffer("vocab_emb_cache", None)

    def precompute_graph(self):
        # We must call this with .detach() if we aren't training GAT
        # or leave it as is if we want the GAT to keep learning.
       self.vocab_emb_cache = self.vocab_gat(self.gat_base_emb.weight, self.adj).detach()

    def forward(self, idx, mode="hybrid"):
        # A. Semantic Features
        x_seq = self.text_enc(idx)
        
        # B. Structural Features
        if self.vocab_emb_cache is not None:
            g_struct = F.embedding(idx, self.vocab_emb_cache)
        else:
            g_struct = torch.zeros_like(x_seq)
            
        
        if mode == "transformer_only":
           
            combined = torch.cat([x_seq, torch.zeros_like(g_struct)], dim=-1)
        elif mode == "gat_only":
            combined = torch.cat([torch.zeros_like(x_seq), g_struct], dim=-1)
        else:
            # Hybrid: Weight them based on the learned gate
            # Use sigmoid to keep gate between 0 and 1
            g = torch.sigmoid(self.gate)
            combined = torch.cat([(1-g)*x_seq, g*g_struct], dim=-1)
        
        x_fused = F.relu(self.fusion_layer(combined))
        
        mask = (idx != 0).unsqueeze(-1)
        x_pooled = (x_fused * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.classifier(x_pooled)