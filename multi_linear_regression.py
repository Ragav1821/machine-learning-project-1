import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df=pd.read_csv("Data_file - data_file.csv")
df.head()
df.info()
lb=LabelEncoder()
df['ocean_proximity1']=lb.fit_transform(df['ocean_proximity'])
df.drop('ocean_proximity', axis=1, inplace=True)
df = df.dropna()
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.show()
sns.pairplot(df)
plt.show()
df.hist(bins=20, figsize=(10, 8))
plt.show()
X = df[["median_income"]]
y = df["median_house_value"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
print("R2 Score:", r2_score(y_test, y_pred))
print(X_test_scaled.shape)
print(y_test.shape)
plt.scatter(y_test, y_pred, color='pink', alpha=0.5)
plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.title("Multiple Linear Regression: Actual vs Predicted")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')  # perfect prediction line
plt.show()
user_input = {}
features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms',
            'population', 'households', 'median_income', 'ocean_proximity1']
for feature in features:
    user_input[feature] = float(input(f"Enter {feature}: "))
user_df = pd.DataFrame([user_input])
user_scaled = scaler.transform(user_df)
predicted_price = model.predict(user_scaled)
predicted_price = max(0, predicted_price[0])
print(f"\n🏠 Predicted House Price: ${predicted_price:,.2f}")






