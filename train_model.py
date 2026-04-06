import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

print("Y-Knot Fraud Detective - Training Started")

# Load data
data = pd.read_csv('data/creditcard.csv')
print(f"Loaded {len(data):,} transactions")
print(f"Fraud cases: {data['Class'].sum()} ({data['Class'].mean()*100:.4f}%)")

# Create images folder
os.makedirs('images', exist_ok=True)

# Class distribution plot
plt.figure(figsize=(8, 5))
sns.countplot(x='Class', data=data)
plt.title('Fraud vs Non-Fraud Transactions')
plt.savefig('images/class_distribution.png')
plt.close()

# Prepare features and target
X = data.drop('Class', axis=1)
y = data['Class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = xgb.XGBClassifier(
    scale_pos_weight=570,
    random_state=42,
    eval_metric='auc'
)

print("Training XGBoost model...")
model.fit(X_train_scaled, y_train)

# Save model and scaler
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/fraud_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

print("Training completed successfully. Model and plots saved.")