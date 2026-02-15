from build_co_occurence import build_coocc
from load import preproc_data
from fusion_model import HumanAIDetector
import torch
import torch.nn.functional as F
from metrics import eval_fusion_ablation,print_full_metrics
device = 'cuda'
block_size = 128
batch_size = 64
vocab_size,itos,train_dataset,val_dataset,train_loader,val_loader = preproc_data(block_size,batch_size,device)

adj_dense = build_coocc(train_dataset,vocab_size=vocab_size,window=5)
indices=(adj_dense>0).nonzero(as_tuple = False).t()  # storing the elements index as tuple hence (_,2)
indices.shape
values = adj_dense[indices[0],indices[1]]
adj_sparse = torch.sparse_coo_tensor(indices,values,
                                     size=adj_dense.shape).coalesce().to(device)

nembed = 64

# 1. Initialize
model = HumanAIDetector(vocab_size, nembed, adj_sparse).to(device)
epochs = 50


optimizer_all = torch.optim.AdamW(model.parameters(), lr=1e-3)

print("Phase 1: Training GAT Only (Transformer Frozen)...")

for epoch in range(epochs):
    model.train()
    
    # Freeze Transformer
    for param in model.text_enc.parameters():
        param.requires_grad = False
    # Ensure GAT and Embeddings are unfrozen
    for param in model.vocab_gat.parameters():
        param.requires_grad = True
    for param in model.gat_base_emb.parameters():
        param.requires_grad = True

    total_loss = 0.0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        
        
        model.vocab_emb_cache = model.vocab_gat(model.gat_base_emb.weight, model.adj)
        
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        
        optimizer_all.zero_grad()
        loss.backward()
        optimizer_all.step()
        
        total_loss += loss.item()
print("Phase 2: Unfreezing Transformer & Fine-tuning Full Model...")


# 1. Unfreeze the text encoder
for param in model.text_enc.parameters():
    param.requires_grad = True


optimizer_fine = torch.optim.AdamW(model.parameters(), lr=1e-5) 

fine_tune_epochs = 10 

for epoch in range(fine_tune_epochs):
    model.train()
    total_loss = 0.0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        
      
        model.vocab_emb_cache = model.vocab_gat(model.gat_base_emb.weight, model.adj)
        
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        
        optimizer_fine.zero_grad()
        loss.backward()
        optimizer_fine.step()
        
        total_loss += loss.item()
    
  
acc_full = eval_fusion_ablation(model, val_loader, device, ablate_gat=False)
acc_transformer_only = eval_fusion_ablation(model, val_loader, device, ablate_gat=True)

print(f"Full Hybrid Model Acc:   {acc_full:.4%}")
print(f"Transformer Only Acc:    {acc_transformer_only:.4%}")
print(f"GAT Contribution:        {(acc_full - acc_transformer_only):.4%}")
print_full_metrics(model, val_loader, device)
    
    