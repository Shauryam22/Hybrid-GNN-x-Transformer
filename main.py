from build_co_occurence import build_coocc
from load import preproc_data
from fusion_model import HumanAIDetector
import torch
import torch.nn.functional as F
device = 'cuda'
block_size = 128
batch_size = 64
vocab_size,train_dataset,val_dataset,train_loader,val_loader = preproc_data(block_size,batch_size,device)

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

# 2. DEFINING OPTIMIZERS
# We will freeze the text encoder partway through
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
        
        # FIX: Re-compute the graph inside the loop so a NEW graph is built for every batch
        # This prevents the "backward through the graph a second time" error.
        model.vocab_emb_cache = model.vocab_gat(model.gat_base_emb.weight, model.adj)
        
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        
        optimizer_all.zero_grad()
        loss.backward()
        optimizer_all.step()
        
        total_loss += loss.item()

# --- START PHASE 2: JOINT FINE-TUNING ---
print("Phase 2: Unfreezing Transformer & Fine-tuning Full Model...")


# 1. Unfreeze the text encoder
for param in model.text_enc.parameters():
    param.requires_grad = True

# 2. Use a smaller learning rate for fine-tuning
# This is crucial so the Transformer doesn't lose its pre-trained "knowledge"
optimizer_fine = torch.optim.AdamW(model.parameters(), lr=1e-5) 

fine_tune_epochs = 10 

for epoch in range(fine_tune_epochs):
    model.train()
    total_loss = 0.0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        
        # RECTIFICATION: 
        # Re-compute graph inside the batch loop so gradients can flow 
        # to the GAT and Transformer simultaneously without a graph-reuse error.
        model.vocab_emb_cache = model.vocab_gat(model.gat_base_emb.weight, model.adj)
        
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        
        optimizer_fine.zero_grad()
        loss.backward()
        optimizer_fine.step()
        
        total_loss += loss.item()
    
  
    
    
    