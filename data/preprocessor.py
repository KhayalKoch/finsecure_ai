import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Ensure processed folder exists
os.makedirs("data/processed", exist_ok=True)

# 1️⃣ Load raw dataset
df = pd.read_csv("data/raw/transactions.csv")
print(f"Raw dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# 2️⃣ Drop unnecessary columns (IDs, timestamps)
df = df.drop(columns=["transaction_id", "timestamp", "sender_id", "receiver_id"])

# 3️⃣ Separate target
y = df["fraud_label"]
X = df.drop(columns=["fraud_label"])

# 4️⃣ Extra feature engineering
# Hour bucket (0-6 night, 6-12 morning, 12-18 afternoon, 18-24 evening)
X["hour_bucket"] = pd.cut(
    X["hour"],
    bins=[-1, 6, 12, 18, 24],
    labels=["night", "morning", "afternoon", "evening"]
)

# Ratio feature: amount / average amount
X["amount_ratio"] = X["amount"] / X["amount"].mean()

# 5️⃣ Define categorical and numeric columns
categorical_features = ["transaction_type", "location", "hour_bucket"]
numeric_features = ["amount", "is_international", "amount_ratio", "hour"]

# 6️⃣ Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(sparse_output=False, drop="first"), categorical_features)
    ]
)

pipeline = Pipeline(steps=[("preprocessor", preprocessor)])

# 7️⃣ Transform features
X_processed = pipeline.fit_transform(X)
print(f"Processed features shape: {X_processed.shape}")

# 8️⃣ Train/test split (stratified, reproducible)
X_train, X_test, y_train, y_test = train_test_split(
    X_processed,
    y,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=y
)

# 9️⃣ Save processed dataset
pd.DataFrame(X_train).to_csv("data/processed/X_train.csv", index=False)
pd.DataFrame(X_test).to_csv("data/processed/X_test.csv", index=False)
pd.DataFrame(y_train).to_csv("data/processed/y_train.csv", index=False)
pd.DataFrame(y_test).to_csv("data/processed/y_test.csv", index=False)

# 🔹 Save preprocessing pipeline (scaler + encoder) for future use
with open("data/processed/preprocessor.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("Processed dataset ready and preprocessor saved as pickle!")

# 10️⃣ Data validation / sanity checks
fraud_ratio = y.mean()
print(f"Fraud ratio: {fraud_ratio:.4%}")
print(f"Train fraud count: {y_train.sum()}, Test fraud count: {y_test.sum()}")