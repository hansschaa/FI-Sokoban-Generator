import optuna
import pandas as pd
import os

db_url = os.environ.get("OPTUNA_DB_URL", "sqlite:///path_consistency.db")
study_name = "path_consistency_optuna"

print(f"Conectando a {db_url}...")
study = optuna.load_study(study_name=study_name, storage=db_url)

# Convertir todos los trials a un DataFrame de Pandas
df = study.trials_dataframe()

# Guardar en un CSV
output_file = "resultados_optuna_parciales.csv"
df.to_csv(output_file, index=False)

print(f"\n¡Exportación completada!")
print(f"Total de trials registrados: {len(df)}")
print(f"Resultados guardados en: {output_file}")

# Mostrar los mejores 3 resultados en la consola:
print("\nTop 3 mejores trials hasta el momento:")
top3 = df[df["state"] == "COMPLETE"].sort_values(by="value", ascending=False).head(3)
if not top3.empty:
    for i, row in top3.iterrows():
        print(f"Trial {row['number']}: Valor = {row['value']:.4f}")
        print(f"  Params: lr={row.get('params_lr', 'N/A')}, alpha={row.get('params_alpha', 'N/A')}, margin={row.get('params_margin', 'N/A')}")
else:
    print("Aún no hay trials completados.")
