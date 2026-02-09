import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import os

# Create images folder
if not os.path.exists('images'):
    os.makedirs('images')

print("--- 1. GENERATING DATA ---")
np.random.seed(42)
n_samples = 3000

data = {
    'Gender': np.random.choice(['Male', 'Female'], n_samples),
    'Age': np.random.randint(18, 75, n_samples),
    'Married': np.random.choice(['Yes', 'No'], n_samples),
    'BankCustomer': np.random.choice(['Yes', 'No'], n_samples),
    'YearsEmployed': np.random.uniform(0, 25, n_samples).round(1),
    'PriorDefault': np.random.choice(['Yes', 'No'], n_samples), 
    'Employed': np.random.choice(['Yes', 'No'], n_samples),
    'CreditScore': np.random.randint(0, 20, n_samples),
    'Income': np.random.randint(0, 100000, n_samples),
    'Approved': np.zeros(n_samples) 
}
df = pd.DataFrame(data)

# Logic for target (Complex rule)
# Approval requires decent credit score + no prior default OR very high income
mask = ((df['CreditScore'] > 4) & (df['PriorDefault'] == 'No')) | (df['Income'] > 40000)
df.loc[mask, 'Approved'] = 1

# Save Data for EDA
df.to_csv('credit_data.csv', index=False)
print("✅ credit_data.csv saved.")

# --- PREPROCESSING ---
print("--- 2. PREPROCESSING ---")
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['Married'] = df['Married'].map({'Yes': 1, 'No': 0})
df['BankCustomer'] = df['BankCustomer'].map({'Yes': 1, 'No': 0})
df['PriorDefault'] = df['PriorDefault'].map({'Yes': 1, 'No': 0})
df['Employed'] = df['Employed'].map({'Yes': 1, 'No': 0})

X = df.drop('Approved', axis=1)
y = df['Approved']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- TRAIN MODEL 1: RANDOM FOREST ---
print("--- 3. TRAINING RANDOM FOREST ---")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf_model.predict(X_test))
joblib.dump(rf_model, 'credit_model_rf.pkl')
print(f"✅ Random Forest Saved (Accuracy: {rf_acc:.2%})")

# --- TRAIN MODEL 2: LOGISTIC REGRESSION ---
print("--- 4. TRAINING LOGISTIC REGRESSION ---")
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_acc = accuracy_score(y_test, lr_model.predict(X_test))
joblib.dump(lr_model, 'credit_model_lr.pkl')
print(f"✅ Logistic Regression Saved (Accuracy: {lr_acc:.2%})")

# --- GENERATE PLOTS ---
print("--- 5. GENERATING INSIGHTS ---")

# Feature Importance (Random Forest)
feature_imp = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_imp, y=feature_imp.index, palette='viridis')
plt.title("What Matters Most for Approval?")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig('images/feature_importance.png')
plt.close()

# Confusion Matrix (Random Forest)
cm = confusion_matrix(y_test, rf_model.predict(X_test))
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f"Confusion Matrix (RF Accuracy: {rf_acc:.0%})")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig('images/confusion_matrix.png')
plt.close()

print("\n🎉 SUCCESS! All models and images are ready.")
input("Press Enter to exit...")
