import pandas as pd

def summarize_optuna(file_path, direction):
    df = pd.read_csv(file_path)
    # Filter completed trials
    completed = df[df['state'] == 'COMPLETE']
    
    if len(completed) == 0:
        print(f"No completed trials in {file_path}")
        return
        
    if direction == "maximize":
        best_idx = completed['value'].idxmax()
    else:
        best_idx = completed['value'].idxmin()
        
    best = completed.loc[best_idx]
    
    print(f"Summary for {file_path}:")
    print(f"Total Trials: {len(df)}")
    print(f"Completed: {len(completed)}")
    print(f"Pruned: {len(df[df['state'] == 'PRUNED'])}")
    print(f"Best Value: {best['value']:.4f}")
    
    # print params
    params = {col: best[col] for col in df.columns if col.startswith('params_')}
    print(f"Best Params: {params}")
    print("-" * 40)

summarize_optuna("optuna_results/optuna_export_sokoban_classifier_lab_v4.csv", "maximize")
summarize_optuna("optuna_results/optuna_export_sokoban_regressor_lab_v4.csv", "minimize")
