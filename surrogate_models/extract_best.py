import optuna
import json
import os

db_url = os.environ.get("OPTUNA_DB_URL", "mysql+pymysql://sokoban:laboratorio123@172.16.16.124/optuna_db")
study_name = os.environ.get("OPTUNA_STUDY_NAME", "sokoban_classifier_lab")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

print(f"Conectando a {db_url} (Estudio: {study_name})...")
study = optuna.load_study(study_name=study_name, storage=db_url)

is_classifier = "classifier" in study_name.lower()

print("\n" + "="*55)
print(f"  RESULTADOS FINALES DEL ENJAMBRE ({'CLASIFICADOR' if is_classifier else 'REGRESOR'})")
print("="*55)
print(f"  Total de Trials globales ejecutados/podados: {len(study.trials)}")

if is_classifier:
    print(f"  Mejor F_0.5 Score: {study.best_value:.4f}")
    out_file = "best_hparams_classifier.json"
else:
    print(f"  Mejor MAE Pushes: {study.best_value:.2f} empujes")
    out_file = "best_hparams.json"

print(f"  Mejores hiperparámetros:")
for k, v in study.best_params.items():
    print(f"    {k}: {v}")

out_path = os.path.join(RESULTS_DIR, out_file)
with open(out_path, "w") as f:
    json.dump({
        "study_name": study_name,
        "best_value": study.best_value,
        "params": study.best_params
    }, f, indent=2)

print(f"\n  ✅ Hiperparámetros guardados en: {out_path}")
