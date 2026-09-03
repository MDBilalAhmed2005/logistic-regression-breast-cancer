import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve
)

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

print("Feature dataset shape:", X.shape)
print("\nTarget labels:")
print(pd.Series(data.target_names))
print("\nClass counts:")
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nTraining data shape:", X_train_scaled.shape)
print("Testing data shape:", X_test_scaled.shape)

# Train logistic regression model
model = LogisticRegression(max_iter=10000, random_state=42)
model.fit(X_train_scaled, y_train)

# Predict on test set
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print("\nModel trained successfully.")
print("First 10 predicted labels:", y_pred[:10])
print("First 10 predicted probabilities:", y_prob[:10])

# Evaluation metrics
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print(f"\nPrecision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

# Confusion matrix plot
plt.figure(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues")
plt.title("Confusion Matrix - Breast Cancer Logistic Regression")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Breast Cancer Logistic Regression")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150)
plt.show()

# Threshold tuning example
best_threshold = 0.5
best_f1 = 0

for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
    y_pred_thresh = (y_prob >= thresh).astype(int)
    p = precision_score(y_test, y_pred_thresh)
    r = recall_score(y_test, y_pred_thresh)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    print(f"Threshold {thresh:.2f} -> Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = thresh

print(f"\nBest threshold by F1: {best_threshold:.2f} (F1 = {best_f1:.4f})")

# Sigmoid explanation
import numpy as np

z = np.linspace(-6, 6, 200)
sigmoid = 1 / (1 + np.exp(-z))

plt.figure(figsize=(5, 4))
plt.plot(z, sigmoid)
plt.axhline(0.5, color="gray", linestyle="--")
plt.xlabel("z (linear combination of features)")
plt.ylabel("P(y=1 | x)")
plt.title("Sigmoid Function Used in Logistic Regression")
plt.tight_layout()
plt.savefig("sigmoid_function.png", dpi=150)
plt.show()

print("\nSigmoid function: sigma(z) = 1 / (1 + exp(-z))")
print("It maps any real z to (0, 1), interpreted as the probability of class 1 (benign).")
print("By default, threshold 0.5 is used: if sigma(z) >= 0.5 -> predict benign, else malignant.")
