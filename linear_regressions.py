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
plt.scatter(X_test_scaled, y_test, color='blue', label='Actual')
plt.plot(X_test_scaled, y_pred, color='red', label='Best fit line')
plt.xlabel("Median Income")
plt.ylabel("House Value")
plt.legend()
plt.show()
features = ['median_income']
user_input = {}
for feature in features:
    user_input[feature] = float(input(f"Enter {feature}: "))
    values = [value for value in user_input.values()]
user_df = pd.DataFrame([values], columns=user_input.keys())

predicted_price = model.predict(user_df.values)

# --- 🖨️ Output ---
print(f"\n🏠 Predicted House Price: ${predicted_price[0]:,.2f}")



#Enter median_income:  4000000

#🏠 Predicted House Price: $316,942,099,827.39










