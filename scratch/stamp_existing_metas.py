"""
Script de migración one-shot: estampa el pipeline_hash actual en los metas
existentes que NO son full_surrogate (que ya sabemos son válidos).
Ejecutar UNA sola vez en el lab antes de relanzar run_exp1_2x2_matrix.py.
"""
import os, sys, json, hashlib

OUTPUT_DIR = "experiment_1_matrix_results"

PIPELINE_HASH_FILES = [
    "./build/experiment_runner",
    "./build2/experiment_runner",
    "surrogate_models/surrogate_server.py",
    "surrogate_models/results/regressor_calibration.json",
    "src/neural_heuristic.cpp",
    "surrogate_models/results/surrogate_stats.txt",
]

def compute_pipeline_hash():
    h = hashlib.sha256()
    for path in PIPELINE_HASH_FILES:
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    h.update(path.encode())
                    h.update(f.read())
            except Exception:
                pass
    return h.hexdigest()[:16]

CURRENT_HASH = compute_pipeline_hash()
print(f"Pipeline Hash actual: {CURRENT_HASH}")

# Solo estampar las variantes que NO son full_surrogate (esas las borramos y re-corremos)
VALID_HEURISTICS = ["hungarian", "classifier_filter", "hybrid_regressor"]

stamped = 0
skipped = 0

for fname in os.listdir(OUTPUT_DIR):
    if not fname.endswith("_meta.json"):
        continue
    
    # Determinar heuristic del nombre del archivo
    heuristic = None
    for h in VALID_HEURISTICS:
        if fname.startswith(h + "_"):
            heuristic = h
            break
    
    if heuristic is None:
        print(f"  ⏭️  Saltando (no válido para estampar): {fname}")
        skipped += 1
        continue
    
    fpath = os.path.join(OUTPUT_DIR, fname)
    try:
        with open(fpath, "r") as f:
            data = json.load(f)
        
        if "pipeline_hash" in data:
            print(f"  ✅ Ya tiene hash ({data['pipeline_hash'][:8]}…): {fname}")
            continue
        
        data["pipeline_hash"] = CURRENT_HASH
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        
        print(f"  🔏 Estampado: {fname}")
        stamped += 1
    except Exception as e:
        print(f"  ❌ Error en {fname}: {e}")

print(f"\nResumen: {stamped} metas estampados, {skipped} saltados.")
print(f"Ahora podés relanzar run_exp1_2x2_matrix.py — solo correrá los full_surrogate.")
