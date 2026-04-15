import pandas as pd
import numpy as np
import pickle
import os
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# 1. Load the processed datasets
print("Loading processed data...")
X_train = pd.read_csv("data/processed/X_train.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

# 2. Strategy for Imbalanced Data: Calculate scale_pos_weight
# Logic: count(negative) / count(positive)
pos_count = np.sum(y_train == 1)
neg_count = np.sum(y_train == 0)
scale_weight = neg_count / pos_count
print(f"Calculated scale_pos_weight: {scale_weight:.2f}")

# 3. Initialize XGBoost Model (Our Baseline 'Horse')
print("Initializing XGBoost Classifier...")
model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_weight,  # The "hidden weapon" against imbalance
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

# 4. K-Fold Cross-Validation (Ensuring the model learns, not memorizes)
print("Performing 5-Fold Cross-Validation...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_f1_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='f1_weighted')
print(f"CV F1-Score (Weighted): {cv_f1_scores.mean():.4f} (+/- {cv_f1_scores.std():.4f})")

# 5. Training the model on full train set
print("Training final baseline model...")
model.fit(X_train, y_train)

# 6. Evaluation on Test Set
y_pred = model.predict(X_test)

# --- METRIC ANALYSIS ---
print("\n" + "="*50)
print("BASELINE MODEL PERFORMANCE (WEEK 1 REPORT)")
print("="*50)

# Accuracy Check (The "Deceptive" Metric)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {acc:.4%}")

# Confusion Matrix (The "Source of Truth")
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Detailed Classification Report (Precision, Recall, F1)
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

# 7. Save the trained model
os.makedirs("model", exist_ok=True)
with open("model/baseline_xgb.pkl", "wb") as f:
    pickle.dump(model, f)

print("\n✅ Baseline model and report are ready!")