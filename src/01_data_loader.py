# Stage 01: Data Loader - HGAFF-R
# Run load_data() to get df_clean for Stage 02

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def load_data(filepath='/content/HGAFF-R/data/paysim.csv'):
    df = pd.read_csv(filepath)
    print(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns")
    df_clean = df.drop(columns=["nameOrig", "nameDest"], errors="ignore")
    return df_clean

if __name__ == "__main__":
    df = load_data()
    print(df.head())
