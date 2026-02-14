import torch
def build_coocc(dataset, vocab_size, window=8): 
    print('Refining Structural Code Graph...')
    
    all_windows = []
    width = window + 1

    for para, _ in dataset:
        para = para[para != 0]  # Remove padding
        if len(para) < width:
            continue
        
        # Create sliding windows to capture broader code context
        w = para.unfold(0, width, 1)
        all_windows.append(w)

    windows = torch.cat(all_windows, dim=0)
    sources = windows[:, 0].repeat_interleave(window)
    targets = windows[:, 1:].flatten()
    
    linear_indices = sources * vocab_size + targets
    uniq_ind, counts = torch.unique(linear_indices, return_counts=True)

    rows = uniq_ind // vocab_size
    cols = uniq_ind % vocab_size

    indices = torch.stack([rows, cols])
    indices_sym = torch.stack([cols, rows])
    all_indices = torch.cat([indices, indices_sym], dim=1)
    
    # LOG-FREQUENCY SCALING:
    # This prevents 'the' or 'public' from dominating the GNN weights.
    all_val = torch.log1p(torch.cat([counts.float(), counts.float()])) 

    adj = torch.sparse_coo_tensor(all_indices, all_val, (vocab_size, vocab_size)).coalesce()
    adj_dense = adj.to_dense()

    # Self-loops and Symmetric Normalization
    adj_dense += torch.eye(vocab_size)
    row_sum = adj_dense.sum(dim=1)
    d_inv_sqrt = torch.pow(row_sum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    
    norm_adj = d_mat_inv_sqrt @ adj_dense @ d_mat_inv_sqrt
    return norm_adj