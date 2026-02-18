import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv("heart dataset.csv")
df['weight'] = df['weight'].astype(int)

df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
df['age_years'] = df['age'] // 365
df['cholesterol'] = df['cholesterol'].replace({1: 0, 2: 1, 3: 1})  # Normal=0, above=1
df['gluc'] = df['gluc'].replace({1: 0, 2: 1, 3: 1})
df['map'] = df['ap_lo'] + (df['pulse_pressure'] / 3)
df['age_bmi'] = df['age_years'] * df['bmi']
df['bp_ratio'] = df['ap_hi'] / (df['ap_lo'] + 1)
df['high_bp'] = ((df['ap_hi'] > 140) | (df['ap_lo'] > 90)).astype(int)
df['obese'] = (df['bmi'] > 30).astype(int)

features = [
    'age', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc',
    'smoke', 'alco', 'active', 'bmi', 'pulse_pressure',
    'map', 'age_bmi', 'bp_ratio', 'high_bp', 'obese'
]
X = df[features]
y = df['disease']  # Continuous numeric target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsRegressor(n_neighbors=30))  # You can tune this!
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("✅ Mean Squared Error:", mse)
print("✅ R² Score:", r2)
