import pandas as pd
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TSV_FILE = os.path.join(SCRIPT_DIR, "../surrogate_models/results/path_consistency_heldout.tsv")

if os.path.exists(TSV_FILE):
    df = pd.read_csv(TSV_FILE, sep='\t')
    print(f"Rows: {len(df)}")
    print(df['Status'].value_counts())
else:
    print("TSV no existe")
