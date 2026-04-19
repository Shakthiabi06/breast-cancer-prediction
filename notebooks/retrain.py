import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Use clean_dataset.csv — raw unscaled values with proper column names
df = pd.read_csv("clean_dataset.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale raw values
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Train Linear SVM
model = LinearSVC(max_iter=10000)
model.fit(X_train_scaled, y_train)

accuracy = model.score(X_test_scaled, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Saved model.pkl and scaler.pkl successfully!")

# Quick test with a known Benign sample
import numpy as np
benign_sample = [[13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,
                  0.1887,0.05916,0.2034,0.7446,1.474,20.1,0.0045,0.009,
                  0.0102,0.00599,0.01211,0.00218,15.11,19.26,99.7,711.2,
                  0.144,0.1773,0.239,0.1288,0.2977,0.07259]]
sample_df = pd.DataFrame(benign_sample, columns=X.columns)
pred = model.predict(scaler.transform(sample_df))[0]
print(f"Benign sample test: {'✓ Benign' if pred == 0 else '✗ Malignant (unexpected)'}")
