"""
export_contrastive_optuna_analysis.py
-------------------------------------
Script para exportar, resumir y analizar en profundidad los resultados del enjambre de Optuna 
del clasificador contrastivo desde la base de datos MySQL del clúster de laboratorio.

Genera:
 1. optuna_contrastive_trials_export.csv con la totalidad de los trials (para gráficas y anexos del paper).
 2. best_hparams_contrastive_classifier.json con los parámetros ganadores y umbral ideal para producción.
 3. Reporte en consola del Top-5 de modelos con desglose de Precisión, Recall, F0.5 y Umbral Óptimo.

Ejecución en cualquiera de las terminales conectadas a la BD (incluso mientras el estudio se está ejecutando o al terminar):
    export OPTUNA_DB_URL="mysql+pymysql://optuna_user:sokoban123@172.22.32.164:3306/optuna_db"
    export OPTUNA_STUDY_NAME="sokoban_contrastive_lab_v2"
    venv/bin/python surrogate_models/export_contrastive_optuna_analysis.py
"""

import os, json
import optuna
import pandas as pd
import numpy as np

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

study_name = os.environ.get("OPTUNA_STUDY_NAME", "sokoban_contrastive_lab_v2")
db_url     = os.environ.get("OPTUNA_DB_URL",     "mysql+pymysql://optuna_user:sokoban123@172.22.32.164:3306/optuna_db")

print(f"{'='*65}")
print("  EXPORTACIÓN Y ANÁLISIS DE OPTUNA — CLASIFICADOR CONTRASTIVO")
print(f"  Base de datos: {db_url.split('@')[-1] if '@' in db_url else db_url}")
print(f"  Estudio: {study_name}")
print(f"{'='*65}\n")

try:
    study = optuna.load_study(study_name=study_name, storage=db_url)
    
    # 1. Exportación a CSV de todos los trials
    df = study.trials_dataframe()
    out_csv = os.path.join(RESULTS_DIR, "optuna_contrastive_trials_export.csv")
    df.to_csv(out_csv, index=False)
    
    total_trials = len(study.trials)
    completed    = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned       = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed       = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    
    print(f"✅ ¡Estudio descargado con éxito!")
    print(f"📊 Estadísticas del Enjambre:")
    print(f"   * Total de trials evaluados: {total_trials}")
    print(f"   * Completados (llegaron al final o early stopping): {len(completed)}")
    print(f"   * Podados tempranamente (MedianPruner): {len(pruned)}")
    print(f"   * Fallidos / Interrumpidos: {len(failed)}")
    print(f"   * Archivo CSV completo guardado en: {out_csv}\n")
    
    if not completed:
        print("⚠️ Aún no hay trials en estado COMPLETE en este estudio. Inténtelo más tarde.")
        exit(0)
        
    # 2. Resumen y extracción de hiperparámetros ganadores
    best_trial = study.best_trial
    stats = best_trial.user_attrs.get("best_stats", {})
    
    print(f"🏆 MEJOR TRIAL DE LA BÚSQUEDA: [Trial #{best_trial.number}]")
    print(f"   * F_0.5 Score Máximo: {best_trial.value:.5f}")
    print(f"   * Calibración (Umbral Óptimo): {stats.get('optimal_threshold', 'N/A')}")
    print(f"   * Precisión @ Umbral: {stats.get('precision', 'N/A')}")
    print(f"   * Recall    @ Umbral: {stats.get('recall', 'N/A')}")
    print(f"   * Época del Óptimo:    {stats.get('epoch', 'N/A')}\n")
    
    print("📋 Hiperparámetros de Entrenamiento Ganadores:")
    for k, v in best_trial.params.items():
        if isinstance(v, float):
            print(f"   * {k:<15}: {v:.6g}")
        else:
            print(f"   * {k:<15}: {v}")
            
    # Guardar JSON con la configuración ganadora lista para train_production_model o similar
    out_json = os.path.join(RESULTS_DIR, "best_hparams_contrastive_classifier.json")
    best_payload = {
        "study_name": study_name,
        "best_trial_number": best_trial.number,
        "best_f_05": best_trial.value,
        "best_params": best_trial.params,
        "optimal_threshold": stats.get("optimal_threshold", 0.70),
        "metrics_at_optimum": stats
    }
    with open(out_json, "w") as f:
        json.dump(best_payload, f, indent=2)
    print(f"\n✅ Configuración óptima guardada para uso en producción en:\n   -> {out_json}\n")

    # 3. Análisis Top-5 para el paper y robustez de calibración
    print(f"{'='*65}")
    print("  TOP-5 MEJORES TRIALS DEL ENJAMBRE (Análisis de Estabilidad)")
    print(f"{'='*65}")
    
    sorted_trials = sorted(completed, key=lambda t: t.value if t.value is not None else -1, reverse=True)[:5]
    print(f"{'Rank':<5} | {'Trial':<6} | {'F_0.5':<8} | {'Umbral':<7} | {'Precisión':<9} | {'Recall':<8} | {'LR':<9} | {'Drop':<5} | {'W_Decay':<8} | {'BS':<4}")
    print("-" * 88)
    for i, t in enumerate(sorted_trials, 1):
        s = t.user_attrs.get("best_stats", {})
        prec = f"{s.get('precision', 0):.4f}" if isinstance(s.get('precision'), (int, float)) else "N/A"
        rec  = f"{s.get('recall', 0):.4f}" if isinstance(s.get('recall'), (int, float)) else "N/A"
        uth  = f"{s.get('optimal_threshold', 'N/A')}"
        lr   = f"{t.params.get('lr', 0):.2e}"
        drop = f"{t.params.get('dropout_p', 0):.2f}"
        wd   = f"{t.params.get('weight_decay', 0):.2e}"
        bs   = f"{t.params.get('batch_size', '')}"
        print(f" #{i:<4} | #{t.number:<5} | {t.value:.5f} | {uth:<7} | {prec:<9} | {rec:<8} | {lr:<9} | {drop:<5} | {wd:<8} | {bs:<4}")
    print("-" * 88)
    
except Exception as e:
    print(f"❌ Error al consultar la base de datos o exportar: {e}")
    import traceback
    traceback.print_exc()
