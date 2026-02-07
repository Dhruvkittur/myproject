import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

print("--- STARTING PROJECT ---")

# 1. GENERATE MOCK DATA
print("1. Generating synthetic data...")
np.random.seed(42)
n_samples = 2000

data = {
    'Gender': np.random.choice(['Male', 'Female'], n_samples),
    'Age': np.random.randint(18, 70, n_samples),
    'Married': np.random.choice(['Yes', 'No'], n_samples),
    'BankCustomer': np.random.choice(['Yes', 'No'], n_samples),
    'YearsEmployed': np.random.uniform(0, 20, n_samples).round(1),
    'PriorDefault': np.random.choice(['Yes', 'No'], n_samples), 
    'Employed': np.random.choice(['Yes', 'No'], n_samples),
    'CreditScore': np.random.randint(0, 20, n_samples),
    'Income': np.random.randint(0, 50000, n_samples),
    'Approved': np.zeros(n_samples) 
}

df = pd.DataFrame(data)

# Logic for target
mask = (df['CreditScore'] > 5) & (df['PriorDefault'] == 'No') & (df['Income'] > 5000)
df.loc[mask, 'Approved'] = 1

# --- NEW STEP: SAVE DATA FOR EDA ---
print("2. Saving data for Dashboard...")
df.to_csv('credit_data.csv', index=False)
print("   ✅ credit_data.csv created!")

# 3. PREPROCESSING
print("3. Preprocessing...")
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['Married'] = df['Married'].map({'Yes': 1, 'No': 0})
df['BankCustomer'] = df['BankCustomer'].map({'Yes': 1, 'No': 0})
df['PriorDefault'] = df['PriorDefault'].map({'Yes': 1, 'No': 0})
df['Employed'] = df['Employed'].map({'Yes': 1, 'No': 0})

X = df.drop('Approved', axis=1)
y = df['Approved']

# 4. TRAIN MODEL
print("4. Training Model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 5. SAVE MODEL
joblib.dump(model, 'credit_model.pkl')
print("   ✅ credit_model.pkl created!")

print("\nSUCCESS! Both files are ready.")
input("Press Enter to exit...")