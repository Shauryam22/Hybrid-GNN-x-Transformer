from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch



def print_full_metrics(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    print("Generating metrics...")
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            
            # Forward pass (uses your concatenation fusion automatically)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1)
            
            # Move to CPU and convert to numpy for sklearn
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())

    # 1. The "All-in-One" Report
    # target_names assumes 0=Human, 1=AI based on your earlier code
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(all_labels, all_preds, target_names=['Human Code', 'AI Code'], digits=4))

    # 2. Confusion Matrix (To see False Positives vs False Negatives)
    cm = confusion_matrix(all_labels, all_preds)
    print("CONFUSION MATRIX")
    print(f"True Negatives (Human correctly ID'd): {cm[0][0]}")
    print(f"False Positives (Human labeled as AI): {cm[0][1]}")
    print(f"False Negatives (AI labeled as Human): {cm[1][0]}")
    print(f"True Positives  (AI correctly ID'd):   {cm[1][1]}")
    print("="*60)
    
    return accuracy_score(all_labels, all_preds)

def eval_fusion_ablation(model, val_loader, device, ablate_gat=False):
    model.eval()
    correct, total = 0, 0
    
    with torch.no_grad():
        for xv, yv in val_loader:
            xv, yv = xv.to(device), yv.to(device)
            
            # 1. Get Transformer Features manually
            x_seq = model.text_enc(xv)
            
            # 2. Get GAT Features manually
            if model.vocab_emb_cache is not None:
                g_struct = torch.nn.functional.embedding(xv, model.vocab_emb_cache)
            else:
                g_struct = torch.zeros_like(x_seq)
            
            # 3. ABLATION LOGIC: If we want to test "Transformer Only", 
            # we force the GAT features to be zeros.
            if ablate_gat:
                g_struct = torch.zeros_like(g_struct)
                
            # 4. Manual Fusion (Concatenate)
            combined = torch.cat([x_seq, g_struct], dim=-1)
            x_fused = torch.nn.functional.relu(model.fusion_layer(combined))
            
            # 5. Pooling & Classify
            mask = (xv != 0).unsqueeze(-1)
            x_pooled = (x_fused * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            logits = model.classifier(x_pooled)
            
            preds = torch.argmax(logits, dim=1)
            correct += (preds == yv).sum().item()
            total += yv.size(0)
            
    return correct / total

