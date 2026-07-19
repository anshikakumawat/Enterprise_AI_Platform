"""Lag Features"""
import pandas as pd

def create_lag_features(df, column):
    df[f"{column}_lag1"] = df[column].shift(1)
    df[f"{column}_lag2"] = df[column].shift(2)
    return df

"""SMOTE"""
from imblearn.over_sampling import SMOTE

def apply_smote(X, y):
    smote = SMOTE(random_state=42, k_neighbors=1)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

"""Target Encoding"""
def target_encoding(df, category_column, target_column):
    target_mean = df.groupby(category_column)[target_column].mean()
    df[category_column] = df[category_column].map(target_mean)
    return df

"""ColumnTransformer"""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def preprocess_pipeline(numerical_columns, categorical_columns):
    transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_columns),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
        ]
    )
    return transformer


