# ============================================================
# STAGE 02: PREPROCESSING + FEATURE ENGINEERING
# HGAFF-R Pipeline
# ============================================================

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import pickle

def preprocess(filepath="/content/HGAFF-R/data/paysim.csv"):

    # Load
    df = pd.read_csv(filepath)
    df = df.drop(columns=["nameOrig", "nameDest", "isFlaggedFraud"],
                 errors="ignore")

    # Encode type
    df = pd.get_dummies(df, columns=["type"], prefix="type", dtype=int)

    # Feature engineering
    df["errorBalanceOrig"]  = df["newbalanceOrig"] - df["oldbalanceOrg"] + df["amount"]
    df["errorBalanceDest"]  = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    df["balanceOrigRatio"]  = df["amount"] / (df["oldbalanceOrg"] + 1)
    df["balanceDestRatio"]  = df["amount"] / (df["oldbalanceDest"] + 1)
    df["balanceChangeOrig"] = df["newbalanceOrig"] - df["oldbalanceOrg"]
    df["balanceChangeDest"] = df["newbalanceDest"] - df["oldbalanceDest"]
    df["isZeroBalanceOrig"] = (df["newbalanceOrig"] == 0).astype(int)
    df["isZeroBalanceDest"] = (df["oldbalanceDest"] == 0).astype(int)
    df["isLargeTransaction"]= (df["amount"] > 200000).astype(int)
    df["isAccountEmptied"]  = ((df["oldbalanceOrg"] > 0) & (df["newbalanceOrig"] == 0)).astype(int)
    df["hourOfDay"]         = df["step"] % 24
    df["dayOfMonth"]        = (df["step"] // 24) + 1
    df["isWeekend"]         = (df["dayOfMonth"] % 7 >= 5).astype(int)

    # Separate X and y
    X = df.drop(columns=["isFraud"])
    y = df["isFraud"]

    # Scale numerical features
    num_cols = [
        "step", "amount",
        "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest",
        "errorBalanceOrig", "errorBalanceDest",
        "balanceOrigRatio", "balanceDestRatio",
        "balanceChangeOrig", "balanceChangeDest",
        "hourOfDay", "dayOfMonth"
    ]
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    # Save outputs
    os.makedirs("/content/HGAFF-R/outputs", exist_ok=True)
    pickle.dump(X,      open("/content/HGAFF-R/outputs/X_preprocessed.pkl", "wb"))
    pickle.dump(y,      open("/content/HGAFF-R/outputs/y_preprocessed.pkl", "wb"))
    pickle.dump(scaler, open("/content/HGAFF-R/outputs/scaler.pkl",         "wb"))

    print(f"X shape : {X.shape}")
    print(f"y shape : {y.shape}")
    print("✅ Stage 02 Complete!")
    return X, y, scaler

if __name__ == "__main__":
    X, y, scaler = preprocess()
