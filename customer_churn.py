import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv("churn data.csv")
df.info()
df = df.drop('customerID', axis=1)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['SeniorCitizen'] = df['SeniorCitizen'].astype(int)
df = df.dropna(subset=['TotalCharges'])
df.info()
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
X = df.drop('Churn', axis=1)
y = df['Churn']
numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = [col for col in X.columns if X[col].dtype == 'object']
numeric_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_pipeline, numeric_features),
    ('cat', categorical_pipeline, categorical_features)
])
model = Pipeline(steps=[
    ('preprocessor', preprocessor),          
    ('classifier', SVC(probability=True))           
])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

expected_inputs = {
    'gender': 'Male',
    'SeniorCitizen': 0,
    'age': 35,
    'tenure': 12,
    'MonthlyCharges': 50.0,
    'TotalCharges': 600.0,
    'Partner': 'No',
    'Dependents': 'No',
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'InternetService': 'DSL',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'No',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check'
}
user_input_data = {}
for field, default in expected_inputs.items():
    value = input(f"{field} (default={default}): ").strip()
    if field in ['SeniorCitizen', 'age', 'tenure']:
        user_input_data[field] = int(value) if value else default
    elif field in ['MonthlyCharges', 'TotalCharges']:
        user_input_data[field] = float(value) if value else default
    else:
        user_input_data[field] = value if value else default
user_input_df = pd.DataFrame([user_input_data])
prediction = model.predict(user_input_df)
probability = model.predict_proba(user_input_df)[0][1]
if prediction[0] == 1:
    print(f"🔴 Churn likely! (Probability: {probability:.2f})")
else:
    print(f"🟢 Not likely to churn. (Probability: {probability:.2f})")
























