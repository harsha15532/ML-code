# ML-code

# ================================
# HEART ATTACK PREDICTION USING RANDOM FOREST
# WITH GRAPHS AND METRICS
# ================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier

# ================================
# LOAD ANY ONE OF THE DATASETS
# ================================

# Dataset 1 (Cleveland + Hungary + Statlog)
df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/heart_statlog_cleveland_hungary_final.csv')

# Dataset 2 (Medical dataset)
# df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Medicaldataset.csv')

# Dataset 3 (Heart attack prediction)
# df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/heart_attack_prediction_dataset 3.csv')

print("Dataset Shape:", df.shape)
display(df.head())


# ================================
# BASIC PREPROCESSING
# ================================
df = df.dropna() # Remove missing values if any

# Identify target column (try to detect automatically)
possible_targets = ['target', 'HeartDisease', 'Outcome', 'output', 'num']
target_col = None

for col in df.columns:
    if col in possible_targets:
        target_col = col

if target_col is None:
    raise ValueError("Target column not found. Please rename your target to 'target'.")

print("Using target:", target_col)

X = df.drop(columns=[target_col])
y = df[target_col]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ================================
# RANDOM FOREST CLASSIFIER
# ================================
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

rf.fit(X_train_scaled, y_train)

y_pred = rf.predict(X_test_scaled)
y_pred_proba = rf.predict_proba(X_test_scaled)[:, 1]


# ================================
# MODEL METRICS
# ================================
print("\n===== MODEL PERFORMANCE =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))


# ================================
# 1️⃣ CORRELATION HEATMAP
# ================================
plt.figure(figsize=(12, 7))
sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


# ================================
# 2️⃣ FEATURE IMPORTANCE GRAPH
# ================================
importances = rf.feature_importances_
features = X.columns

plt.figure(figsize=(12, 7))
sns.barplot(x=importances, y=features, palette="viridis")
plt.title("Feature Importance from Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()


# ================================
# 3️⃣ CONFUSION MATRIX HEATMAP
# ================================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# ================================
# 4️⃣ ROC CURVE
# ================================
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()
