import pandas as pd
from sklearn.linear_model import LogisticRegression
import csv
from collections import defaultdict
import warnings

def main():
    warnings.filterwarnings('ignore')
    
    csv_file = "benchmark_results_0_to_40.csv"
    meta_file = "benchmark_stratified_heldout_meta.csv"
    
    # ---------------------------------------------------------
    # 1. INTERSECCIÓN Y MÉTRICAS GLOBALES
    # ---------------------------------------------------------
    data = defaultdict(dict)
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                board_id = int(row['board_id'])
                heur = row['heuristic']
                data[board_id][heur] = row
    except FileNotFoundError:
        print(f"Error: No se encontro {csv_file}")
        return

    heuristics = ['manhattan', 'hungarian', 'neural_sequential', 'neural_batched_massive']
    
    print("="*105)
    print("1. RENDIMIENTO GLOBAL EN LA INTERSECCIÓN")
    print("="*105)
    
    intersection = []
    for board_id, runs in data.items():
        if all(heur in runs and runs[heur]['status'] == 'SOLVED' for heur in heuristics):
            intersection.append(board_id)
            
    print(f"Tableros resueltos por TODAS las heurísticas (Intersección): {len(intersection)} / 40\n")
    print(f"{'Heurística':<25} | {'Resueltos':<10} | {'Nodos (Intersección)':<25} | {'Tiempo (Intersección)'}")
    print("-" * 105)
    
    for heur in heuristics:
        solved_total = sum(1 for b in data.values() if heur in b and b[heur]['status'] == 'SOLVED')
        
        inter_nodes = 0
        inter_time = 0
        for b_id in intersection:
            inter_nodes += int(data[b_id][heur]['expanded_nodes'])
            inter_time += float(data[b_id][heur]['runtime_ms'])
            
        avg_nodes = inter_nodes / len(intersection) if intersection else 0
        avg_time = inter_time / len(intersection) if intersection else 0
        
        print(f"{heur:<25} | {solved_total:>4} / 40 | {avg_nodes:>10.1f} nodos promedio | {avg_time:>10.1f} ms promedio")

    # ---------------------------------------------------------
    # 2. ANÁLISIS DETALLADO DE FALLOS Y VARIABLES
    # ---------------------------------------------------------
    try:
        meta_df = pd.read_csv(meta_file)
        if 'index' in meta_df.columns:
            meta_df = meta_df.rename(columns={'index': 'board_id'})
    except Exception as e:
        print("\nError loading metadata:", e)
        return

    try:
        results_df = pd.read_csv(csv_file)
    except Exception as e:
        print("\nError loading results:", e)
        return
        
    pivot_df = results_df.pivot(index='board_id', columns='heuristic', values='status')
    df = pd.merge(meta_df, pivot_df, on='board_id', how='inner')
    
    for h in heuristics:
        if h in df.columns:
            df[f'{h}_solved'] = df[h] == 'SOLVED'
            
    print("\n" + "="*105)
    print("2. TASA DE RESOLUCIÓN POR BOX_COUNT")
    print("="*105)
    
    box_col = 'box_count' if 'box_count' in df.columns else ('boxes' if 'boxes' in df.columns else None)
    pushes_col = 'gt_pushes' if 'gt_pushes' in df.columns else ('pushes' if 'pushes' in df.columns else None)
    bucket_col = 'bucket' if 'bucket' in df.columns else None

    if box_col:
        for h in heuristics:
            if f'{h}_solved' in df.columns:
                print(f"\n{h}:")
                print(df.groupby(box_col)[f'{h}_solved'].agg(['mean', 'count']))
    else:
        print("No se encontro columna de box_count.")
                
    print("\n" + "="*105)
    print("3. REGRESIÓN LOGÍSTICA (Pesos de Variables)")
    print("="*105)
    
    if box_col and pushes_col:
        X_cols = [box_col, pushes_col]
        if all(c in df.columns for c in X_cols):
            X = df[X_cols]
            y = df['neural_batched_massive_solved']
            
            try:
                model = LogisticRegression(max_iter=1000).fit(X, y)
                print("Coeficientes de LogisticRegression para predecir exito de 'neural_batched_massive':")
                for col, coef in zip(X.columns, model.coef_[0]):
                    print(f"  - {col}: {coef:.4f}")
                print("* Nota: Un coeficiente negativo fuerte indica que a mayor valor, cae la probabilidad de resolver.")
            except Exception as e:
                print("No se pudo ajustar el modelo:", e)
    else:
        print("Columnas de metadata no encontradas para Regresión Logística.")
        
    print("\n" + "="*105)
    print("4. DESGLOSE DE TABLEROS FALLIDOS (Hungarian = SOLVED, Neural Massive = FAILED)")
    print("="*105)
    
    failed_mask = (df['hungarian_solved'] == True) & (df['neural_batched_massive_solved'] == False)
    failed_df = df[failed_mask]
    
    print(f"Total tableros en esta condicion: {len(failed_df)}\n")
    
    if not failed_df.empty:
        print(f"{'Board':<7} | {'Meta (Cajas / Pushes)':<25} | {'Hungarian (Nodos -> ms)':<25} | {'Massive Fallido (Nodos)':<25} | {'Tipo de Fallo'}")
        print("-" * 115)
        for _, row in failed_df.iterrows():
            bid = row['board_id']
            b = row.get(box_col, '?')
            p = row.get(pushes_col, '?')
            bucket = row.get(bucket_col, '?')
            
            h_row = results_df[(results_df['board_id'] == bid) & (results_df['heuristic'] == 'hungarian')].iloc[0]
            n_row = results_df[(results_df['board_id'] == bid) & (results_df['heuristic'] == 'neural_batched_massive')].iloc[0]
            
            h_n = int(h_row['expanded_nodes'])
            h_t = float(h_row['runtime_ms'])
            n_n = int(n_row['expanded_nodes'])
            n_t = float(n_row['runtime_ms'])
            
            if float(n_n) < float(h_n) * 0.5:
                tipo = "Lentitud Computacional (Timeout)"
            else:
                tipo = "Explosion de Nodos / Mala Guia"
                
            meta_str = f"C: {b} | P: {p} ({bucket})"
            print(f"{bid:<7} | {meta_str:<25} | {h_n:>6} n -> {h_t:>5.0f} ms | {n_n:>6} n -> {n_t:>5.0f} ms | {tipo}")

    print("\n" + "="*105)
    print("5. OPTIMALIDAD DE LA SOLUCIÓN (Comparación de Pushes en Intersección)")
    print("="*105)
    
    equal_count = 0
    suboptimal_count = 0
    superoptimal_count = 0
    suboptimal_boards = []

    if intersection:
        for b_id in intersection:
            p_hungarian = int(data[b_id]['hungarian']['pushes'])
            p_neural = int(data[b_id]['neural_batched_massive']['pushes'])
            diff = p_neural - p_hungarian
            
            if diff == 0:
                equal_count += 1
            elif diff > 0:
                suboptimal_count += 1
                # Retrieve box_count from the dataframe
                try:
                    box_c = df.loc[df['board_id'] == b_id, box_col].iloc[0]
                except:
                    box_c = '?'
                suboptimal_boards.append({'board_id': b_id, 'diff': diff, 'box_count': box_c, 'p_hung': p_hungarian, 'p_neur': p_neural})
            else:
                superoptimal_count += 1
                
        print(f"Total de tableros evaluados (Intersección Estricta): {len(intersection)}")
        print(f" (a) Igual de Óptima (Mismos pushes) : {equal_count} tableros ({(equal_count/len(intersection))*100:.1f}%)")
        print(f" (b) Subóptima       (Más pushes)    : {suboptimal_count} tableros ({(suboptimal_count/len(intersection))*100:.1f}%)")
        print(f" (c) Superóptima     (Menos pushes)  : {superoptimal_count} tableros ({(superoptimal_count/len(intersection))*100:.1f}%)")
        
        if suboptimal_count > 0:
            print("\nDesglose de tableros con solución subóptima (Neural > Hungarian):")
            print(f"{'Board':<7} | {'Cajas':<7} | {'Hungarian Pushes':<18} | {'Neural Pushes':<15} | {'Diferencia (Diff)'}")
            print("-" * 75)
            for b in suboptimal_boards:
                print(f"{b['board_id']:<7} | {b['box_count']:<7} | {b['p_hung']:<18} | {b['p_neur']:<15} | +{b['diff']}")
    else:
        print("No hay tableros en la intersección.")

if __name__ == "__main__":
    main()
