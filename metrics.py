from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

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

# Run it
print_full_metrics(model, val_loader, device)