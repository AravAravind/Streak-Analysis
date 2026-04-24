import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

# 1. LOAD AND PREPROCESS
print("Loading data...")
df = pd.read_csv('cities_weather_maharashtra.csv')
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

# 2. DATA CLEANING (Median Imputation)
print("Cleaning data...")
cols_to_fix = ['tavg', 'tmin', 'tmax', 'prcp', 'pres']
for col in cols_to_fix:
    df[col] = df.groupby(['city', 'month'])[col].transform(lambda x: x.fillna(x.median()))

# Drop columns with 100% missing values (wspd) and others not needed for core ML
df = df.drop(columns=['wspd', 'tsun'], errors='ignore')
df.dropna(subset=['tavg'], inplace=True)

# 3. FANCY VISUALIZATIONS
sns.set_theme(style="whitegrid")

# Chart A: Temperature Heatmap
plt.figure(figsize=(14, 8))
pivot_df = df.groupby(['city', 'month'])['tavg'].mean().unstack()
sns.heatmap(pivot_df, cmap='YlOrRd', annot=False)
plt.title('Heatmap: Average Temperature by City and Month', fontsize=16)
plt.show()

# Chart B: Rainfall Ridge Plot (Conceptual)
plt.figure(figsize=(12, 6))
sns.violinplot(data=df, x='month', y='prcp', palette='Blues', inner="quart")
plt.yscale('log')
plt.title('Rainfall Intensity Distribution (Log Scale)', fontsize=16)
plt.ylabel('Precipitation (mm)')
plt.show()

# 4. MACHINE LEARNING
print("Training Machine Learning Model...")
le = LabelEncoder()
df['city_code'] = le.fit_transform(df['city'])

X = df[['year', 'month', 'city_code']]
y = df['tavg']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print(f"--- Model Performance ---")
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f}°C")

# 5. PREDICTION EXAMPLE (Predicting May 2026 for a city)
test_city = "Mumbai"
city_encoded = le.transform([test_city])[0]
future_pred = model.predict([[2026, 5, city_encoded]])
print(f"\nPredicted Avg Temp for {test_city} in May 2026: {future_pred[0]:.2f}°C")