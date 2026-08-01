import torch
import numpy as np
import os
import sys
from sklearn.metrics import fbeta_score, precision_score, recall_score

sys.path.append('surrogate_models')
from models.resnet import SokobanSEResNetClassifier
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cuda')

results = []
for fold in range(1, 6):
    model = SokobanSEResNetClassifier(dropout_p=0.4, in_channels=12).to(device)
    model.load_state_dict(torch.load(f"surrogate_models/results/final_contrastive_classifier_fold{fold}.pt", map_location=device, weights_only=False))
    model.eval()
    
    X_test = torch.load(f"surrogate_models/results/contrastive_fold_{fold-1}_X_test.pt", map_location='cpu')
    y_test = torch.load(f"surrogate_models/results/contrastive_fold_{fold-1}_y_test.pt", map_location='cpu')
    t_test = torch.load(f"surrogate_models/results/contrastive_fold_{fold-1}_t_test.pt", map_location='cpu')
    
    dataset = TensorDataset(X_test, y_test, t_test)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    all_probs = []
    all_targets = []
    all_types = []
    with torch.no_grad():
        for X, y, t in loader:
            probs = torch.sigmoid(model(X.to(device)))
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(y.numpy())
            all_types.extend(t.numpy())
            
    y_true = np.array(all_targets)
    y_prob = np.array(all_probs)
    t_arr = np.array(all_types)
    
    best_f05 = 0
    best_thresh = 0.5
    for thresh in np.arange(0.50, 0.96, 0.05):
        y_pred = (y_prob >= thresh).astype(float)
        f05 = fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = thresh
            
    y_pred = (y_prob >= best_thresh).astype(float)
    mask_simple = (t_arr == 2)
    mask_complex = (t_arr == 3)
    
    spec_simple = (y_pred[mask_simple] == 0).sum() / mask_simple.sum()
    spec_complex = (y_pred[mask_complex] == 0).sum() / mask_complex.sum()
    
    results.append((spec_simple, spec_complex))
    print(f"Fold {fold}: Spec(Simple)={spec_simple:.4f}, Spec(Complex)={spec_complex:.4f}")

spec_s = [r[0] for r in results]
spec_c = [r[1] for r in results]
print(f"Mean Spec(Simple): {np.mean(spec_s):.4f} +/- {np.std(spec_s):.4f}")
print(f"Mean Spec(Complex): {np.mean(spec_c):.4f} +/- {np.std(spec_c):.4f}")
