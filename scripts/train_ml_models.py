import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load and prepare data
print("📄 Loading dataset with embeddings and labels...")
df = pd.read_csv("data/processed/embedded_emails.csv")

print("🧹 Cleaning labels...")
df = df.dropna(subset=["label"])

print("🔍 Extracting features and labels...")
X = df[[f"embedding_{i}" for i in range(768)]].values
y = df["label"].values

print("✂️ Splitting dataset (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print("📏 Scaling features (StandardScaler)...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Random Forest
print("\n🚀 Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print("🧪 Evaluating Random Forest...")
y_pred_rf = rf.predict(X_test)
print(f"✅ Accuracy: {accuracy_score(y_test, y_pred_rf)}")
print("📊 Classification Report:\n", classification_report(y_test, y_pred_rf))
print("🧩 Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
joblib.dump(rf, "models/random_forest_model.pkl")

# Logistic Regression
print("\n🚀 Training Logistic Regression...")
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
print("🧪 Evaluating Logistic Regression...")
y_pred_lr = log_reg.predict(X_test)
print(f"✅ Accuracy: {accuracy_score(y_test, y_pred_lr)}")
print("📊 Classification Report:\n", classification_report(y_test, y_pred_lr))
print("🧩 Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
joblib.dump(log_reg, "models/log_reg_model.pkl")

# SVM
print("\n🚀 Training SVM (RBF kernel)...")
svm = SVC(kernel="rbf")
svm.fit(X_train, y_train)
print("🧪 Evaluating SVM (RBF kernel)...")
y_pred_svm = svm.predict(X_test)
print(f"✅ Accuracy: {accuracy_score(y_test, y_pred_svm)}")
print("📊 Classification Report:\n", classification_report(y_test, y_pred_svm))
print("🧩 Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))
joblib.dump(svm, "models/svm_model.pkl")

print("\n💾 All models trained and saved successfully in /models folder!")
