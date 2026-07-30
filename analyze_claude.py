import pandas as pd
from sklearn.linear_model import LogisticRegression
import sys
import warnings

def main():
    # Silenciar warnings de sklearn
    warnings.filterwarnings('ignore')
    
    csv_file = "benchmark_results_0_to_40.csv"
    meta_file = "benchmark_stratified_heldout_meta.csv"
    
    # Cargar metadatos
    try:
        meta_df = pd.read_csv(meta_file)
        if 'index' in meta_df.columns:
            meta_df = meta_df.rename(columns={'index': 'board_id'})
    except Exception as e:
        print("Error loading metadata:", e)
        return

    # Cargar resultados
    try:
        results_df = pd.read_csv(csv_file)
    except Exception as e:
        print("Error loading results:", e)
        return
        
    pivot_df = results_df.pivot(index='board_id', columns='heuristic', values='status')
    df = pd.merge(meta_df, pivot_df, on='board_id', how='inner')
    
    heuristics = ['manhattan', 'hungarian', 'neural_sequential', 'neural_batched_massive']
    
    for h in heuristics:
        if h in df.columns:
            df[f'{h}_solved'] = df[h] == 'SOLVED'
            
    print("="*80)
    print("1. TASA DE RESOLUCIÓN POR BOX_COUNT")
    print("="*80)
    
    box_col = 'boxes' if 'boxes' in df.columns else 'box_count'
    pushes_col = 'pushes' if 'pushes' in df.columns else 'gt_pushes'
    
    for h in heuristics:
        if f'{h}_solved' in df.columns:
            print(f"\n{h}:")
            print(df.groupby(box_col)[f'{h}_solved'].agg(['mean', 'count']))
                
    print("\n" + "="*80)
    print("2. REGRESIÓN LOGÍSTICA (Pesos de Variables)")
    print("="*80)
    
    X_cols = [box_col, pushes_col]
    
    if all(c in df.columns for c in X_cols):
        X = df[X_cols]
        y = df['neural_batched_massive_solved']
        
        try:
            model = LogisticRegression(max_iter=1000).fit(X, y)
            print("Coeficientes de LogisticRegression para predecir 'neural_batched_massive_solved':")
            for col, coef in zip(X.columns, model.coef_[0]):
                print(f"  - {col}: {coef:.4f}")
            print("\n* Nota: Un coeficiente negativo fuerte indica que a mayor valor, cae la probabilidad de resolver.")
        except Exception as e:
            print("No se pudo ajustar el modelo:", e)
    else:
        print("Columnas de metadata no encontradas para Regresión Logística.")
        
    print("\n" + "="*80)
    print("3. DESGLOSE DE TABLEROS FALLIDOS (Hungarian = SOLVED, Neural Massive = FAILED)")
    print("="*80)
    
    failed_mask = (df['hungarian_solved'] == True) & (df['neural_batched_massive_solved'] == False)
    failed_df = df[failed_mask]
    
    print(f"Total tableros en esta condición: {len(failed_df)}\n")
    
    if not failed_df.empty:
        for _, row in failed_df.iterrows():
            bid = row['board_id']
            b = row.get(box_col, '?')
            p = row.get(pushes_col, '?')
            
            h_row = results_df[(results_df['board_id'] == bid) & (results_df['heuristic'] == 'hungarian')].iloc[0]
            n_row = results_df[(results_df['board_id'] == bid) & (results_df['heuristic'] == 'neural_batched_massive')].iloc[0]
            
            h_n = int(h_row['expanded_nodes'])
            h_t = float(h_row['runtime_ms'])
            n_n = int(n_row['expanded_nodes'])
            n_t = float(n_row['runtime_ms'])
            
            if float(n_n) < float(h_n) * 0.5:
                tipo = "Lentitud Computacional (Timeout limite)"
            else:
                tipo = "Explosion de Nodos / Mala Guia"
                
            bucket = "91+" if int(p) >= 91 else ("Alta" if int(p) >= 70 else "Media")
                
            print(f"Board {bid:<2} | Cajas: {b}, Bucket: {bucket:<4} ({p} p) | Hung: {h_n:>6}n ({h_t:>5.0f}ms) | Neur: {n_n:>6}n ({n_t:>5.0f}ms) | {tipo}")

if __name__ == "__main__":
    main()
