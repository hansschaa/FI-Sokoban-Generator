import optuna
import json
import os

db_url = os.environ.get("OPTUNA_DB_URL", "mysql+pymysql://sokoban:laboratorio123@172.16.16.124/optuna_db")
study_name = os.environ.get("OPTUNA_STUDY_NAME", "sokoban_regressor_lab")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

print(f"Conectando a {db_url}...")
study = optuna.load_study(study_name=study_name, storage=db_url)

print("\n" + "="*55)
print("  RESULTADOS FINALES DEL ENJAMBRE")
print("="*55)
print(f"  Total de Trials ejecutados/podados: {len(study.trials)}")
print(f"  Mejor MAE Pushes: {study.best_value:.2f} empujes")
print(f"  Mejores hiperparámetros:")
for k, v in study.best_params.items():
    print(f"    {k}: {v}")

out_path = os.path.join(RESULTS_DIR, "best_hparams.json")
with open(out_path, "w") as f:
    json.dump({"best_mae": study.best_value, "params": study.best_params}, f, indent=2)
print(f"\n  ✅ Hiperparámetros guardados en: {out_path}")
