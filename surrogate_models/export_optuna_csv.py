import optuna
import os
import pandas as pd

# Puedes cambiar este nombre por el estudio que quieras exportar
study_name = os.getenv("OPTUNA_STUDY_NAME", "regressor_seresnet_v1")
db_url = os.getenv("OPTUNA_DB_URL", "mysql+pymysql://sokoban:laboratorio123@172.16.16.124/optuna_db")

print(f"Conectando a {db_url}...")
print(f"Cargando estudio: {study_name}")

try:
    study = optuna.load_study(study_name=study_name, storage=db_url)
    
    # Exportar los trials a un dataframe de Pandas
    df = study.trials_dataframe()
    
    # Guardar en un CSV
    out_file = f"optuna_export_{study_name}.csv"
    df.to_csv(out_file, index=False)
    
    print(f"¡Exportación exitosa! Se encontraron {len(df)} trials.")
    print(f"Resumen guardado en: {out_file}")
    
    # Imprimir los mejores si existe al menos 1 trial terminado
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) > 0:
        print("\nMejor trial hasta el momento:")
        print(f"  Valor (MAE): {study.best_value:.4f}")
        print(f"  Parámetros:  {study.best_params}")
    else:
        print("\nAún no hay ningún trial completo (estado COMPLETE).")

except KeyError:
    print(f"Error: No se encontró el estudio '{study_name}' en la base de datos.")
    print("Asegúrate de que el nombre sea el correcto (v1 o v2).")
