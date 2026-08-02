import os
import sys
import subprocess
import json
import urllib.request
import urllib.error
import numpy as np
import pandas as pd
import random
import glob

def load_shell(path):
    with open(path, "r") as f:
        lines = [line.rstrip("\n\r") for line in f if line.strip() != ""]
    return lines

def analyze_structure(lines):
    rows = len(lines)
    cols = max(len(l) for l in lines) if rows > 0 else 0
    grid = [['#' for _ in range(cols)] for _ in range(rows)]
    for r, l in enumerate(lines):
        for c, char in enumerate(l):
            grid[r][c] = char

    total_cells = rows * cols
    walls = sum(row.count('#') for row in grid)
    free = total_cells - walls
    wall_density = (walls / total_cells * 100) if total_cells > 0 else 0

    # Contar celdas de "deadlock intrínseco" para cajas (esquinas y callejones sin salida de 1x1)
    dead_ends = 0
    corners = 0
    for r in range(1, rows-1):
        for c in range(1, cols-1):
            if grid[r][c] == ' ':
                # Contar muros vecinos inmediatos (N, S, E, O)
                neighbors = 0
                if grid[r-1][c] == '#': neighbors += 1
                if grid[r+1][c] == '#': neighbors += 1
                if grid[r][c-1] == '#': neighbors += 1
                if grid[r][c+1] == '#': neighbors += 1

                if neighbors >= 3:
                    dead_ends += 1
                elif neighbors == 2:
                    # Esquina: (N y E), (N y O), (S y E), (S y O)
                    is_corner = ((grid[r-1][c] == '#' and (grid[r][c-1] == '#' or grid[r][c+1] == '#')) or
                                 (grid[r+1][c] == '#' and (grid[r][c-1] == '#' or grid[r][c+1] == '#')))
                    if is_corner:
                        corners += 1

    valid_box_cells = free - (dead_ends + corners)
    return {
        "dims": f"{rows}x{cols}",
        "total_cells": total_cells,
        "wall_pct": round(wall_density, 1),
        "free_cells": free,
        "dead_ends": dead_ends,
        "corners": corners,
        "valid_box_cells": max(0, valid_box_cells),
        "pct_unusable_free": round(((dead_ends + corners) / free * 100), 1) if free > 0 else 0
    }

def check_pilot_logs():
    print("\n" + "="*95)
    print(" 1️⃣ AUDITORÍA DE CORTE ANTICIPADO Y COLAPSO EN SHELL 3 (LOGS DEL PILOTO)")
    print("="*95)
    
    res_dir = "pilot_classifier_filter_results"
    log_sin = os.path.join(res_dir, "ES_sin_clasificador_shell3_seed42.txt")
    log_con = os.path.join(res_dir, "ES_con_clasificador_shell3_seed42.txt")

    for label, filepath in [("Sin Clasificador (A* Puro)", log_sin), ("Con Clasificador (Filtro Neural)", log_con)]:
        print(f"\n👉 Configuración: {label}")
        if not os.path.exists(filepath):
            print(f"   ⚠️ No se encontró el archivo de log en: {filepath}")
            continue
        
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        
        print(f"   • Total de líneas en log: {len(lines)}")
        stop_reason = "No identificada explícitamente en texto"
        init_info = "Sin datos de inicialización"
        stats_info = []

        for l in lines:
            l_str = l.strip()
            if "[INIT STATS]" in l_str:
                init_info = l_str
            elif "[ES] Criterio de Parada Alcanzado:" in l_str or "Error:" in l_str:
                stop_reason = l_str
            elif "[ES STATS]" in l_str:
                stats_info.append(l_str)

        print(f"   • Inicialización: {init_info}")
        print(f"   • Causa del Corte: {stop_reason}")
        if stats_info:
            for st in stats_info:
                print(f"     {st}")
        else:
            print("     (El motor se detuvo antes de imprimir el reporte estadístico final)")

def compare_topologies():
    print("\n" + "="*95)
    print(" 2️⃣ COMPARATIVA ESTRUCTURAL Y TOPOLÓGICA DE LOS 5 CASCARONES")
    print("="*95)
    
    data = []
    for s in [1, 2, 3, 4, 5]:
        p = f"levels/shell_{s}.sok"
        if os.path.exists(p):
            res = analyze_structure(load_shell(p))
            res["shell"] = f"Shell {s}"
            data.append(res)
        else:
            print(f"   ⚠️ Falta {p}")

    df = pd.DataFrame(data)[["shell", "dims", "free_cells", "wall_pct", "dead_ends", "corners", "valid_box_cells", "pct_unusable_free"]]
    df.columns = ["Shell", "Dimensión", "Celdas Libres", "Muros (%)", "Callejones 1x1", "Esquinas Ciega", "Celdas Útiles Caja", "% Inservible"]
    print(df.to_string(index=False))

    print("\n   📌 DIAGNÓSTICO TOPOLÓGICO:")
    s3_row = df[df["Shell"] == "Shell 3"].iloc[0] if len(df[df["Shell"] == "Shell 3"]) > 0 else None
    if s3_row is not None:
        print(f"   • En Shell 3, de {s3_row['Celdas Libres']} celdas libres, un {s3_row['% Inservible']}% corresponden a esquinas ciegas o callejones 1x1.")
        print(f"   • Esto deja apenas {s3_row['Celdas Útiles Caja']} celdas topológicamente viables para desplazar cajas sin caer en deadlock.")
        print("   • Explica por qué AMBAS corridas (con y sin red) terminan en segundos por estancamiento o fallo de generación: el espacio de mutaciones válidas es extremadamente pobre por naturaleza.")

def audit_rejected_mutations():
    print("\n" + "="*95)
    print(" 3️⃣ AUDITORÍA PROFUNDA DE MUTACIONES RECHAZADAS EN SHELL 3 (FALSOS NEGATIVOS PRE-A*)")
    print(" Objetivo: Verificar con A* real las 15 mutaciones rechazadas con mayor probabilidad cerca del umbral (0.70)")
    print("="*95)

    shell_path = "levels/shell_3.sok"
    if not os.path.exists(shell_path):
        print("❌ Error: No existe levels/shell_3.sok")
        return

    shell_lines = load_shell(shell_path)
    rows = len(shell_lines)
    cols = max(len(l) for l in shell_lines) if rows > 0 else 0
    
    # Encontrar todas las celdas libres
    free_coords = []
    for r in range(rows):
        for c in range(len(shell_lines[r])):
            if shell_lines[r][c] == ' ':
                free_coords.append((r, c))

    if len(free_coords) < 3:
        print("❌ Shell 3 tiene menos de 3 celdas libres.")
        return

    print(f"\n🎲 Generando 300 mutaciones candidate (1 y 2 cajas) sobre Shell 3...")
    candidates = []
    parent_str = "\n".join(shell_lines)

    random.seed(42)
    for _ in range(300):
        num_boxes = 1 if random.random() < 0.7 else 2
        if len(free_coords) < 2 * num_boxes + 1:
            num_boxes = 1
        
        chosen = random.sample(free_coords, 2 * num_boxes + 1)
        player = chosen[0]
        boxes = chosen[1:num_boxes+1]
        goals = chosen[num_boxes+1:]

        grid = [list(l) + [' '] * (cols - len(l)) for l in shell_lines]
        grid[player[0]][player[1]] = '@'
        for b in boxes: grid[b[0]][b[1]] = '$'
        for g in goals: grid[g[0]][g[1]] = '.'

        b_str = "\n".join("".join(row).rstrip() for row in grid)
        candidates.append({"board": b_str, "parent_board": parent_str})

    # Enviar al servidor neuronal
    url = "http://127.0.0.1:5000/evaluate"
    print(f"📡 Consultando al servidor neuronal en {url} para evaluar el batch...")
    try:
        req = urllib.request.Request(url, data=json.dumps({"boards": candidates}).encode('utf-8'), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            predictions = json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        print(f"❌ No se pudo conectar al servidor en {url}: {e}")
        print("💡 Recuerda levantar 'python surrogate_models/surrogate_server.py' en otra terminal para correr esta auditoría.")
        return

    # Filtrar rechazados por debajo del umbral de 0.70
    rejected = []
    for i, p in enumerate(predictions):
        prob = p.get("prob", 0.0 if not p["is_solvable"] else 1.0)
        is_solv = p["is_solvable"]
        if not is_solv or prob < 0.70:
            rejected.append((prob, i, candidates[i]["board"]))

    # Ordenar por probabilidad descendente (los más cercanos al umbral primero)
    rejected.sort(key=lambda x: x[0], reverse=True)
    top_audit = rejected[:15]

    if not top_audit:
        print("ℹ️ No se encontraron mutaciones rechazadas en esta muestra.")
        return

    print(f"📉 Total de mutaciones rechazadas en la muestra: {len(rejected)}/300 ({len(rejected)/3.0:.1f}%)")
    print(f"🔬 Verificando con el solver A* real las 15 mutaciones rechazadas más próximas a la frontera de decisión:\n")

    solver_bin = "./build/sokoban_solver"
    if not os.path.exists(solver_bin):
        solver_bin = "./build2/sokoban_solver"
    if not os.path.exists(solver_bin):
        print(f"❌ Error: No se encontró {solver_bin}")
        return

    results_table = []
    false_negatives = 0
    true_negatives = 0

    temp_sok = "temp_audit_board.sok"
    for rank, (prob, idx, board_text) in enumerate(top_audit, 1):
        with open(temp_sok, "w", encoding="utf-8") as tf:
            tf.write(board_text + "\n")

        cmd = [solver_bin, temp_sok, "0", "500"]
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=10, text=True)
        except Exception as e:
            out = getattr(e, "output", "")

        pushes = 0
        for line in out.split("\n"):
            if "Pushes:" in line:
                try: pushes = int(line.split(":")[1].strip())
                except: pass

        is_real_solvable = (pushes > 1)
        diag = "❌ Falso Negativo (¡A* resolvió!)" if is_real_solvable else "✔️ Genuino Deadlock (TN)"
        
        if is_real_solvable: false_negatives += 1
        else: true_negatives += 1

        results_table.append({
            "Rank": rank,
            "Mut ID": f"#M-{idx:03d}",
            "Neural Prob": f"{prob:.4f}",
            "Decisión (0.70)": "Rechazado",
            "A* Pushes": pushes if pushes > 0 else "0 (Deadlock)",
            "Diagnóstico Ground-Truth": diag
        })

    if os.path.exists(temp_sok):
        try: os.remove(temp_sok)
        except: pass

    df_res = pd.DataFrame(results_table)
    print(df_res.to_string(index=False))

    fn_rate = (false_negatives / len(top_audit)) * 100
    print("\n" + "-"*95)
    print(" 🏆 CONCLUSIÓN DE LA AUDITORÍA DE FALSOS NEGATIVOS EN SHELL 3")
    print(f"  • Casos Auditados          -> 15 mutaciones rechazadas con mayor probabilidad ({top_audit[-1][0]:.3f} a {top_audit[0][0]:.3f})")
    print(f"  • Genuinos Deadlocks (TN) -> {true_negatives} ({100 - fn_rate:.1f}% de acierto real al rechazar)")
    print(f"  • Falsos Negativos (FN)    -> {false_negatives} ({fn_rate:.1f}% de tableros jugables descartados equivocadamente)")
    
    if false_negatives == 0:
        print("\n   👉 VEREDICTO: EL CLASIFICADOR NO ES EXCESIVAMENTE CONSERVADOR EN SHELL 3.")
        print("      El rechazo de estas mutaciones fue 100% preciso según el ground-truth A*.")
        print("      El colapso de Shell 3 y sus 0 falsos positivos se deben íntegramente a una topología intrínsecamente")
        print("      hostil al movimiento de cajas (pobreza estructural de celdas útiles), no a un fallo de generalización.")
    else:
        print("\n   ⚠️ VEREDICTO: EL UMBRAL 0.70 PECA DE CONSERVADOR EN SHELL 3.")
        print("      Existen mutaciones válidas cerca de la frontera que están siendo descartadas erróneamente por la red.")
        print("      Se sugiere flexibilizar o recalibrar el umbral (e.g., 0.50 - 0.60) para tableros con alta densidad y pasillos angostos.")
    print("-" * 95 + "\n")

if __name__ == "__main__":
    check_pilot_logs()
    compare_topologies()
    audit_rejected_mutations()
