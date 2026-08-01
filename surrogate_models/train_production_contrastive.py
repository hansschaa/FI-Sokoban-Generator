import sys, os, json, random, hashlib, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score

from models.resnet import SokobanSEResNetClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 15

def compute_sha256(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

class ProductionContrastiveDataset(Dataset):
    def __init__(self, X, y, t, is_train=True):
        self.X = X
        self.y = y
        self.t = t
        self.is_train = is_train
        self.print_count = 0

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.is_train:
            # Augmentation D4 on-the-fly aplicada a los 12 canales en simultáneo
            k = random.randint(0, 3)
            flip = random.choice([True, False])
            x = torch.rot90(x, k, [1, 2])
            if flip:
                x = torch.flip(x, [2])
            if self.print_count < 3:
                print(f"   [Visual Check D4] Muestra {idx}: Rotación={k*90}°, Flip={flip} (a los 12 canales en simultáneo)")
                self.print_count += 1
        return x, self.y[idx], self.t[idx]

def load_optuna_hparams():
    hparams_path = os.path.join(RESULTS_DIR, "best_hparams_contrastive_classifier.json")
    if os.path.exists(hparams_path):
        with open(hparams_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        params = config.get("best_params", {})
        batch_size   = int(params.get("batch_size", 128))
        dropout_p    = float(params.get("dropout_p", 0.160201))
        lr           = float(params.get("lr", 0.00233982))
        weight_decay = float(params.get("weight_decay", 0.00397114))
        opt_thresh   = float(config.get("optimal_threshold", 0.80))
        print(f"✅ Hiperparámetros Óptimos Optuna v3 Cargados ({hparams_path}):")
        print(f"   BS={batch_size} | Dropout={dropout_p:.6f} | LR={lr:.2e} | WD={weight_decay:.2e} | Threshold={opt_thresh:.2f}\n")
        return batch_size, dropout_p, lr, weight_decay, opt_thresh
    else:
        print(f"⚠️ {hparams_path} no encontrado. Usando valores óptimos v3 explícitos por defecto.")
        return 128, 0.16020136753305936, 0.00233981881881855, 0.003971139414594639, 0.80

def main():
    print(f"=== ENTRENAMIENTO DE MODELO DE PRODUCCIÓN CONTRASTIVO (100% DATASET) ===")
    print(f"Dispositivo de entrenamiento: {device}")

    batch_size, dropout_p, lr, weight_decay, threshold = load_optuna_hparams()

    # Cargar 100% de los datos sin reservar fold de test (Unión de Fold 0 Train + Test)
    train_X_path = os.path.join(RESULTS_DIR, "contrastive_fold_0_X_train.pt")
    train_y_path = os.path.join(RESULTS_DIR, "contrastive_fold_0_y_train.pt")
    train_t_path = os.path.join(RESULTS_DIR, "contrastive_fold_0_t_train.pt")
    test_X_path  = os.path.join(RESULTS_DIR, "contrastive_fold_0_X_test.pt")
    test_y_path  = os.path.join(RESULTS_DIR, "contrastive_fold_0_y_test.pt")
    test_t_path  = os.path.join(RESULTS_DIR, "contrastive_fold_0_t_test.pt")

    if not all(os.path.exists(p) for p in [train_X_path, train_y_path, test_X_path, test_y_path]):
        print("❌ No se encontraron las particiones de fold 0 en results/ para reconstruir el dataset 100%.")
        return

    print("Cargando y unificando 100% del dataset contrastivo...")
    X_train_fold0 = torch.load(train_X_path, map_location='cpu')
    y_train_fold0 = torch.load(train_y_path, map_location='cpu')
    t_train_fold0 = torch.load(train_t_path, map_location='cpu')
    X_test_fold0  = torch.load(test_X_path, map_location='cpu')
    y_test_fold0  = torch.load(test_y_path, map_location='cpu')
    t_test_fold0  = torch.load(test_t_path, map_location='cpu')

    X_all = torch.cat([X_train_fold0, X_test_fold0], dim=0)
    y_all = torch.cat([y_train_fold0, y_test_fold0], dim=0)
    t_all = torch.cat([t_train_fold0, t_test_fold0], dim=0)
    
    total_samples = len(y_all)
    num_pos = (y_all == 1).sum().item()
    num_neg = (y_all == 0).sum().item()
    print(f"📊 Total Muestras Unificadas: {total_samples:,} (Positivos [Solubles]: {num_pos:,}, Negativos [Deadlocks]: {num_neg:,})")

    train_ds = ProductionContrastiveDataset(X_all, y_all, t_all, is_train=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = SokobanSEResNetClassifier(dropout_p=dropout_p, in_channels=12).to(device)
    pos_weight = torch.tensor([num_neg / max(1, num_pos)]).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"\n🚀 Iniciando entrenamiento por {EPOCHS} épocas con CosineAnnealingLR...")
    start_time = time.time()
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch, _ in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        current_lr = scheduler.get_last_lr()[0]
        print(f"   Epoch [{epoch+1:02d}/{EPOCHS}] | Loss: {avg_loss:.4f} | LR: {current_lr:.2e}")

    elapsed = time.time() - start_time
    print(f"\n✅ Entrenamiento completado en {elapsed:.2f} segundos.")

    # Guardar modelo de producción
    output_model_path = os.path.join(RESULTS_DIR, "production_contrastive_classifier.pt")
    torch.save(model.state_dict(), output_model_path)
    sha256 = compute_sha256(output_model_path)
    print(f"\n💾 Modelo final de producción guardado exitosamente en:")
    print(f"   -> {output_model_path}")
    print(f"   -> SHA256 Checksum: {sha256}")

    # Verificación Final de Desempeño (Full Dataset Evaluation sin Augmentación)
    print(f"\n🔍 Realizando evaluación final de cordura sobre el 100% de las muestras con Umbral={threshold:.2f}...")
    eval_ds = ProductionContrastiveDataset(X_all, y_all, t_all, is_train=False)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size*2, shuffle=False, num_workers=4)
    
    model.eval()
    all_probs, all_targets, all_types = [], [], []
    with torch.no_grad():
        for X_b, y_b, t_b in eval_loader:
            X_b = X_b.to(device)
            logits = model(X_b)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_targets.extend(y_b.numpy())
            all_types.extend(t_b.numpy())

    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    all_types = np.array(all_types)
    preds = (all_probs >= threshold).astype(int)

    acc = accuracy_score(all_targets, preds)
    prec = precision_score(all_targets, preds, zero_division=0)
    rec = recall_score(all_targets, preds, zero_division=0)
    f05 = fbeta_score(all_targets, preds, beta=0.5, zero_division=0)

    # Specificity by deadlock type
    simple_mask = (all_types == 2)
    complex_mask = (all_types == 3)
    
    spec_simple = np.mean(preds[simple_mask] == 0) if np.sum(simple_mask) > 0 else 0.0
    spec_complex = np.mean(preds[complex_mask] == 0) if np.sum(complex_mask) > 0 else 0.0

    print(f"\n📊 [Métricas Finales en 100% del Dataset (Umbral={threshold:.2f})]:")
    print(f"   • Exactitud (Accuracy):      {acc:.4f}")
    print(f"   • Precisión (Precision):     {prec:.4f}")
    print(f"   • Sensibilidad (Recall):     {rec:.4f}")
    print(f"   • F0.5 Score:                {f05:.4f}")
    print(f"   • Especificidad (Simple):    {spec_simple:.4f} (N={np.sum(simple_mask):,})")
    print(f"   • Especificidad (Complex):   {spec_complex:.4f} (N={np.sum(complex_mask):,})")
    print(f"\n🎯 Listo para ser consumido por surrogate_server.py en producción!")

if __name__ == "__main__":
    main()
