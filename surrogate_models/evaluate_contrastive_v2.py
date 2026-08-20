"""
PASO 3 — Evaluación post-entrenamiento del Clasificador Contrastivo v2.
F0.5 separado por tipo de deadlock, percentiles por clase real.

PREREQUISITO: Haber completado train_production_contrastive_v2.py

Uso:
  cd ~/hans/FI-Sokoban-Generator
  source venv/bin/activate
  python3 surrogate_models/evaluate_contrastive_v2.py

  # Para evaluar un checkpoint específico:
  python3 surrogate_models/evaluate_contrastive_v2.py ruta/al/checkpoint.pt
"""
import sys, os, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from sklearn.metrics import fbeta_score, precision_score, recall_score
from models.resnet import SokobanSEResNetClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_CHECKPOINT = "production_contrastive_classifier_v2_combined.pt"


def compute_sha256(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def report(probs, targets, name, thresholds=[0.50, 0.60, 0.70, 0.80, 0.90, 0.95]):
    pos = probs[targets == 1]
    neg = probs[targets == 0]

    print(f"\n{'=' * 65}")
    print(f"  {name} ({len(targets):,} muestras)")
    print(f"{'=' * 65}")

    if len(pos) > 0:
        print(f"  Positivos (n={len(pos):,}):")
        print(f"    Media={np.mean(pos):.4f} | p5={np.percentile(pos, 5):.4f} | "
              f"p10={np.percentile(pos, 10):.4f} | p25={np.percentile(pos, 25):.4f} | "
              f"p50={np.median(pos):.4f} | p75={np.percentile(pos, 75):.4f} | "
              f"p90={np.percentile(pos, 90):.4f} | p95={np.percentile(pos, 95):.4f}")

    if len(neg) > 0:
        print(f"  Negativos (n={len(neg):,}):")
        print(f"    Media={np.mean(neg):.4f} | p5={np.percentile(neg, 5):.4f} | "
              f"p10={np.percentile(neg, 10):.4f} | p25={np.percentile(neg, 25):.4f} | "
              f"p50={np.median(neg):.4f} | p75={np.percentile(neg, 75):.4f} | "
              f"p90={np.percentile(neg, 90):.4f} | p95={np.percentile(neg, 95):.4f}")

    print(f"\n  {'Umbral':<8} | {'F0.5':<8} | {'Prec':<8} | {'Rec':<8}")
    print(f"  {'-' * 42}")
    for th in thresholds:
        preds = (probs >= th).astype(float)
        f05 = fbeta_score(targets, preds, beta=0.5, zero_division=0)
        prec = precision_score(targets, preds, zero_division=0)
        rec = recall_score(targets, preds, zero_division=0)
        print(f"  {th:<8.2f} | {f05:<8.4f} | {prec:<8.4f} | {rec:<8.4f}")


def main():
    # Determinar checkpoint
    if len(sys.argv) > 1:
        ckpt_path = sys.argv[1]
    else:
        ckpt_path = os.path.join(RESULTS_DIR, DEFAULT_CHECKPOINT)

    if not os.path.exists(ckpt_path):
        print(f"❌ No se encontró: {ckpt_path}")
        print(f"   Corré train_production_contrastive_v2.py primero.")
        return

    # ── Verificar checkpoint ──────────────────────────────────────────────────
    print("=" * 65)
    print("  EVALUACIÓN — Clasificador Contrastivo v2")
    print("=" * 65)

    sha256 = compute_sha256(ckpt_path)
    size_mb = os.path.getsize(ckpt_path) / 1e6
    print(f"\n  Checkpoint: {os.path.basename(ckpt_path)}")
    print(f"  Tamaño:     {size_mb:.1f} MB")
    print(f"  SHA-256:    {sha256}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    in_channels = ckpt["stem.0.weight"].shape[1]
    print(f"  in_channels: {in_channels}")

    if in_channels != 12:
        print(f"  ❌ ERROR: Este no es un clasificador contrastivo (esperado 12, got {in_channels}).")
        return

    # ── Cargar modelo ─────────────────────────────────────────────────────────
    model = SokobanSEResNetClassifier(dropout_p=0.10, in_channels=12).to(device)
    model.load_state_dict(ckpt)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parámetros: {n_params:,}")

    # ── Cargar datos de test ──────────────────────────────────────────────────
    print(f"\n📦 Cargando tensores de test (fold 0)...")
    X = torch.load(os.path.join(RESULTS_DIR, "contrastive_fold_0_X_test.pt"), map_location="cpu", weights_only=False)
    y = torch.load(os.path.join(RESULTS_DIR, "contrastive_fold_0_y_test.pt"), map_location="cpu", weights_only=False)
    t = torch.load(os.path.join(RESULTS_DIR, "contrastive_fold_0_t_test.pt"), map_location="cpu", weights_only=False)
    
    s_path = os.path.join(RESULTS_DIR, "contrastive_fold_0_s_test.pt")
    if os.path.exists(s_path):
        s = torch.load(s_path, map_location="cpu", weights_only=False)
    else:
        s = None

    print(f"  X shape: {X.shape} | y: {len(y):,} | pos={int((y==1).sum()):,} | neg={int((y==0).sum()):,}")
    print(f"  t==1: {int((t==1).sum()):,} | t==2: {int((t==2).sum()):,} | t==3: {int((t==3).sum()):,}")
    if s is not None:
        print(f"  s==0 (Original): {int((s==0).sum()):,} | s==1 (Denso): {int((s==1).sum()):,}")

    # ── Inferencia ────────────────────────────────────────────────────────────
    print(f"\n🔍 Corriendo inferencia en {device}...")
    all_probs = []
    batch_size = 256
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            X_batch = X[i:i + batch_size].to(device)
            logits = model(X_batch)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.extend(probs)

    all_probs = np.array(all_probs)
    targets = y.numpy().flatten()
    types = t.numpy().flatten()
    sources = s.numpy().flatten() if s is not None else None

    # ── Reportes ──────────────────────────────────────────────────────────────
    report(all_probs, targets, "GLOBAL")
    
    if sources is not None:
        m_orig = sources == 0
        m_dense_src = sources == 1
        
        if m_orig.sum() > 0:
            report(all_probs[m_orig], targets[m_orig], "ORIGINAL CORPUS (s==0)")
            
        if m_dense_src.sum() > 0:
            report(all_probs[m_dense_src], targets[m_dense_src], "DENSE CORPUS (s==1)")

    # Por tipo de deadlock
    m_solvable = types == 1
    m_simple = types == 2
    m_complex = types == 3
    m_dense = (types == 2) | (types == 3)

    if m_solvable.sum() > 0:
        report(all_probs[m_solvable], targets[m_solvable],
               "SOLVABLE TYPE (t==1)")

    if m_simple.sum() > 0:
        report(all_probs[m_simple], targets[m_simple],
               "SIMPLE DEADLOCK (t==2, corpus denso)")

    if m_complex.sum() > 0:
        report(all_probs[m_complex], targets[m_complex],
               "COMPLEX DEADLOCK (t==3, corpus denso)")

    if m_dense.sum() > 0:
        report(all_probs[m_dense], targets[m_dense],
               "TODOS DEADLOCKS DENSOS (t==2 + t==3)")

    # ── Comparación con modelo anterior ───────────────────────────────────────
    old_path = os.path.join(RESULTS_DIR, "production_contrastive_classifier.pt")
    if os.path.exists(old_path):
        old_sha = compute_sha256(old_path)
        print(f"\n📋 Referencia — Modelo anterior (solo corpus original):")
        print(f"   Archivo: production_contrastive_classifier.pt")
        print(f"   SHA-256: {old_sha}")
        print(f"   (Para comparar, corré este mismo script apuntando al checkpoint viejo)")

    print(f"\n{'=' * 65}")
    print(f"  Evaluación completa.")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
