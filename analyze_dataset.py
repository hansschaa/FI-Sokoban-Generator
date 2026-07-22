import json, pathlib

cells = []

def code_cell(src):
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src}

def md_cell(src):
    return {"cell_type":"markdown","metadata":{},"source":src}

# ── Celda 1: imports + parser ──────────────────────────────────────────────────
cells.append(md_cell("# Dataset Sokoban — Análisis de Tableros Únicos\nBusca recursivamente todos los `.sok` y cuenta tableros únicos por cubeta y en total."))

cells.append(code_cell("""\
import pathlib, re, hashlib
from collections import defaultdict
import pandas as pd

# ─── PARSER ───────────────────────────────────────────────────────────────────
def parse_sok_files(root="."):
    \"\"\"
    Busca recursivamente todos los .sok dentro de 'root'.
    Cada entrada tiene el formato:
        [hash] - pushes:N moves:N ... (línea de stats)
        [filas del tablero]
        [línea vacía]
    Retorna lista de dicts con los campos parseados + el string del tablero.
    \"\"\"
    records = []
    root = pathlib.Path(root)
    sok_files = sorted(root.rglob("*.sok"))
    print(f"Archivos .sok encontrados: {len(sok_files)}")
    for f in sok_files:
        print(f"  → {f}")

    for fpath in sok_files:
        bucket = fpath.stem           # ej: "21_to_30"
        text   = fpath.read_text(encoding="utf-8", errors="ignore")
        blocks = [b.strip() for b in text.split("\\n\\n") if b.strip()]

        for block in blocks:
            lines = block.splitlines()
            if not lines:
                continue

            header = lines[0]
            board_lines = lines[1:]

            # Parsear stats del header: "hash - pushes:N moves:N ..."
            stats = {}
            m = re.match(r"(\\d+)\\s*-\\s*(.*)", header)
            if not m:
                continue
            file_hash = m.group(1)
            stats_str = m.group(2).strip()

            # Extraer todos los pares clave:valor
            for token in stats_str.split():
                if ":" in token:
                    k, v = token.split(":", 1)
                    try:
                        stats[k] = float(v) if "." in v else int(v)
                    except ValueError:
                        stats[k] = v

            # Solo procesar si tiene pushes (formato nuevo con stats completas)
            if "pushes" not in stats and not stats_str.isdigit():
                # Formato viejo: "hash - N" donde N es solo pushes
                try:
                    stats["pushes"] = int(stats_str)
                except:
                    continue
            elif stats_str.isdigit():
                stats["pushes"] = int(stats_str)

            board_str = "\\n".join(board_lines)
            if not board_str:
                continue

            # Hash del tablero (para detectar duplicados exactos)
            board_hash = hashlib.sha256(board_str.encode()).hexdigest()

            records.append({
                "file_hash"  : file_hash,
                "board_hash" : board_hash,
                "bucket"     : bucket,
                "source_file": str(fpath),
                "board"      : board_str,
                **stats
            })

    return pd.DataFrame(records)

# Cambiar esta ruta si tus .sok están en otra carpeta
DATASET_ROOT = "sokoban_dataset_buckets"

df_all = parse_sok_files(DATASET_ROOT)
print(f"\\nTotal de entradas leídas (incluyendo posibles duplicados): {len(df_all)}")
"""))

# ── Celda 2: deduplicación y conteo ───────────────────────────────────────────
cells.append(code_cell("""\
# Deduplicar por hash del tablero (sha256 del string completo del tablero)
df_unique = df_all.drop_duplicates(subset="board_hash").copy()

print("=" * 55)
print(f"  TABLEROS TOTALES LEÍDOS  : {len(df_all):>6}")
print(f"  TABLEROS DUPLICADOS      : {len(df_all) - len(df_unique):>6}")
print(f"  TABLEROS ÚNICOS          : {len(df_unique):>6}")
print("=" * 55)
"""))

# ── Celda 3: resumen por cubeta ────────────────────────────────────────────────
cells.append(code_cell("""\
# Orden natural de las cubetas
def bucket_sort_key(name):
    m = re.match(r"(\\d+)", name)
    return int(m.group(1)) if m else 9999

summary = (df_unique.groupby("bucket")
           .agg(
               tableros_unicos=("board_hash", "count"),
               pushes_min=("pushes", "min"),
               pushes_max=("pushes", "max"),
               pushes_avg=("pushes", "mean"),
           )
           .reset_index()
           .sort_values("bucket", key=lambda s: s.map(bucket_sort_key)))

summary["pushes_avg"] = summary["pushes_avg"].round(1)
summary.columns = ["Cubeta", "Únicos", "Pushes Min", "Pushes Max", "Pushes Avg"]
display(summary)
print(f"\\nTOTAL ÚNICOS EN TODOS LOS BUCKETS: {summary['Únicos'].sum()}")
"""))

# ── Celda 4: gráfico ──────────────────────────────────────────────────────────
cells.append(code_cell("""\
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(summary["Cubeta"], summary["Únicos"], color="steelblue", edgecolor="black")
ax.axhline(y=1000, color="red", linestyle="--", linewidth=1.2, label="Meta (1000)")
ax.set_title("Tableros Únicos por Cubeta de Dificultad", fontsize=14, fontweight="bold")
ax.set_xlabel("Cubeta (empujes)")
ax.set_ylabel("Cantidad de tableros únicos")
ax.tick_params(axis="x", rotation=30)
ax.legend()
plt.tight_layout()
plt.show()
"""))

# ── Celda 5: archivos fuente ──────────────────────────────────────────────────
cells.append(code_cell("""\
print("Archivos .sok procesados y cuántos tableros únicos aporta cada uno:\\n")
src_summary = (df_unique.groupby("source_file")["board_hash"]
               .count()
               .sort_values(ascending=False)
               .to_frame("Únicos"))
display(src_summary)
"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
        "language_info": {"name":"python","version":"3.10.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

out = pathlib.Path("/home/hanss/FI-sokoban-generator/analisis_tableros_unicos.ipynb")
out.write_text(json.dumps(nb, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Notebook escrito en: {out}")
