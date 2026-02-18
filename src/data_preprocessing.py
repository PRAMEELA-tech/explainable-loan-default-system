
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler

def remove_high_missing(df, threshold=0.65):
    missing_percent = df.isnull().mean()
    cols_to_drop = missing_percent[missing_percent > threshold].index
    return df.drop(columns=cols_to_drop)

def impute_missing(df):
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df[cat_cols] = df[cat_cols].fillna("Unknown")
    return df

def log_transform(df):
    financial_cols = [
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY"
    ]
    for col in financial_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col])
    return df

def engineer_features(df):
    new_features = pd.DataFrame()

    new_features["CREDIT_TO_INCOME"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    new_features["ANNUITY_TO_INCOME"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    new_features["AGE_YEARS"] = df["DAYS_BIRTH"] / -365
    new_features["EMPLOYMENT_YEARS"] = df["DAYS_EMPLOYED"] / -365

    df = pd.concat([df, new_features], axis=1)

    return df


def encode_features(df):
    return pd.get_dummies(df, drop_first=True)

def scale_features(X, save_path="/teamspace/studios/this_studio/models/standard_scaler.joblib"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, save_path)
    return X_scaled

def full_preprocessing_pipeline(csv_path):
    df = pd.read_csv(csv_path)
    df = remove_high_missing(df)
    df = impute_missing(df)
    df = log_transform(df)
    df = engineer_features(df)
    df = encode_features(df)
    X = df.drop("TARGET", axis=1)
    y = df["TARGET"]
    X_scaled = scale_features(X)
    return X_scaled, y
