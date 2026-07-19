import pandas as pd
import numpy as np

from core_pipeline.transformation_pipes import (
    create_lag_features,
    target_encoding,
    apply_smote,
    preprocess_pipeline
)

# ----------------------------
# Sample Data
# ----------------------------

data = {
    "Sales": [100, 120, 150, 180, 200],
    "Category": ["A", "B", "A", "B", "A"],
    "Target": [1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

# ----------------------------
# Lag Features
# ----------------------------

print("===== Lag Features =====")
lag_df = create_lag_features(df.copy(), "Sales")
print(lag_df)

# ----------------------------
# Target Encoding
# ----------------------------

print("\n===== Target Encoding =====")
encoded_df = target_encoding(df.copy(), "Category", "Target")
print(encoded_df)

# ----------------------------
# Column Transformer
# ----------------------------

print("\n===== Column Transformer =====")

preprocessor = preprocess_pipeline(
    numerical_columns=["Sales"],
    categorical_columns=["Category"]
)

print(preprocessor)

# ----------------------------
# SMOTE
# ----------------------------

print("\n===== SMOTE =====")

X = np.array([
    [100],
    [120],
    [150],
    [180],
    [200]
])

y = np.array([0, 0, 0, 1, 1])

X_resampled, y_resampled = apply_smote(X, y)

print("Resampled X:")
print(X_resampled)

print("Resampled y:")
print(y_resampled)