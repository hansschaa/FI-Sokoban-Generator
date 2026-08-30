import os, subprocess, json, time
import numpy as np

SHELLS = [1, 2, 3, 4, 5]
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
CORES = 24
TIMEOUT = 300

from run_mini_test import verify_board_with_astar

def run():
    print("=========================================================")
    print(" INICIANDO DIAGNÓSTICO DE PROBABILIDADES (GTX4)")
    print("=========================================================")
    
    deadlock_probs = []
    solved_probs = []
    total_logged = 0
    
    for shell in SHELLS:
        for seed in SEEDS:
            cmd = f"./build/experiment_runner ES FO1 {seed} levels/shell_{shell}.sok --heuristic full_surrogate --timeLimit {TIMEOUT}"
            env = os.environ.copy()
            env["USE_SURROGATE"] = "true"
            env["USE_CLASSIFIER_FILTER"] = "false"
            env["CLASSIFIER_THRESHOLD"] = "0.7" if shell != 3 else "0.6"
            
            p = subprocess.run(cmd, shell=True, env=env, capture_output=True, text=True)
            
            top_boards = []
            for line in p.stdout.split('\n') + p.stderr.split('\n'):
                if line.startswith("RANK_"):
                    parts = line.split(";")
                    if len(parts) >= 4:
                        try:
                            s_prob = float(parts[2].strip())
                            b_str = parts[3].strip()
                            top_boards.append((s_prob, b_str))
                        except: pass
            
            for s_prob, b_str in top_boards:
                total_logged += 1
                _, is_sol, is_inc = verify_board_with_astar(b_str)
                if is_sol:
                    solved_probs.append(s_prob)
                elif not is_inc:
                    deadlock_probs.append(s_prob)
                    
            print(f"[OK] Shell {shell} Seed {seed} analizado.")
            
    print("\n=========================================================")
    print(" 📊 REPORTE FINAL DE DISTRIBUCIONES DE PROBABILIDAD")
    print("=========================================================")
    print(f"Total de individuos de élite auditados: {total_logged}")
    
    def print_stats(name, data):
        if not data:
            print(f"\n[{name}] No hay datos.")
            return
        print(f"\n[{name}] (N={len(data)})")
        print(f"  Media:   {np.mean(data):.4f}")
        print(f"  Mediana: {np.median(data):.4f}")
        print(f"  Mínimo:  {np.min(data):.4f}")
        print(f"  Máximo:  {np.max(data):.4f}")
        
        bins = [0.7, 0.8, 0.9, 0.95, 0.99, 1.01]
        labels = ["[0.70-0.80)", "[0.80-0.90)", "[0.90-0.95)", "[0.95-0.99)", "[0.99-1.00]"]
        counts, _ = np.histogram(data, bins=bins)
        for label, count in zip(labels, counts):
            print(f"  Banda {label}: {count} tableros")

    print_stats("DEADLOCK_Genuino", deadlock_probs)
    print_stats("SOLVED", solved_probs)

if __name__ == "__main__":
    run()
