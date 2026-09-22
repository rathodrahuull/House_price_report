# House Price Prediction - Washington State Real Estate Analysis
# Project: Predicting house prices based on various features
# Student: Rahul Rathod
# Dataset: 4,600+ house records from Washington state
# Objective: Build a machine learning model to predict house prices and analyze market trends

# 1. Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("Libraries imported successfully!")

# 2. Load and Explore Data
df = pd.read_csv('data.csv')

print("Dataset Shape:", df.shape)
print("\nFirst Few Rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())

# 3. Statistical Summary
print("Statistical Summary:")
print(df.describe())

print("\n=== PRICE ANALYSIS ===")
print(f"Average Price: ${df['price'].mean():,.2f}")
print(f"Median Price: ${df['price'].median():,.2f}")
print(f"Min Price: ${df['price'].min():,.2f}")
print(f"Max Price: ${df['price'].max():,.2f}")
print(f"Price Range: ${df['price'].max() - df['price'].min():,.2f}")

# 4. Data Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

axes[0, 0].hist(df['price'], bins=50, color='skyblue', edgecolor='black')
axes[0, 0].set_title('Price Distribution', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Price ($)')
axes[0, 0].set_ylabel('Frequency')

axes[0, 1].scatter(df['bedrooms'], df['price'], alpha=0.5, color='green')
axes[0, 1].set_title('Bedrooms vs Price', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Number of Bedrooms')
axes[0, 1].set_ylabel('Price ($)')

axes[1, 0].scatter(df['sqft_living'], df['price'], alpha=0.5, color='orange')
axes[1, 0].set_title('Living Area vs Price', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Square Feet (Living Area)')
axes[1, 0].set_ylabel('Price ($)')

axes[1, 1].scatter(df['yr_built'], df['price'], alpha=0.5, color='red')
axes[1, 1].set_title('Year Built vs Price', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Year Built')
axes[1, 1].set_ylabel('Price ($)')

plt.tight_layout()
plt.show()
print("Visualizations created successfully!")

# 5. Data Preprocessing
df_processed = df.copy()
df_processed = df_processed.drop(['street', 'country', 'date'], axis=1)
df_processed = df_processed.fillna(df_processed.mean(numeric_only=True))

le_city = LabelEncoder()
le_state = LabelEncoder()

df_processed['city'] = le_city.fit_transform(df_processed['city'])
df_processed['statezip'] = le_state.fit_transform(df_processed['statezip'])

print("Data preprocessing completed!")
print(f"\nProcessed Data Shape: {df_processed.shape}")
print(f"\nProcessed Data Columns: {list(df_processed.columns)}")

# 6. Correlation Analysis
correlation = df_processed.corr()['price'].sort_values(ascending=False)
print("Features Correlation with Price:")
print(correlation)

plt.figure(figsize=(10, 8))
sns.heatmap(df_processed.corr(), annot=False, cmap='coolwarm', center=0)
plt.title('Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# 7. Feature Selection and Model Preparation
X = df_processed.drop('price', axis=1)
y = df_processed['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training Set Size: {X_train.shape[0]}")
print(f"Testing Set Size: {X_test.shape[0]}")
print(f"Number of Features: {X_train.shape[1]}")

# 8. Model Building - Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

y_pred_train_lr = lr_model.predict(X_train_scaled)
y_pred_test_lr = lr_model.predict(X_test_scaled)

print("=== LINEAR REGRESSION ===")
print(f"R² Score (Training): {r2_score(y_train, y_pred_train_lr):.4f}")
print(f"R² Score (Testing): {r2_score(y_test, y_pred_test_lr):.4f}")
print(f"RMSE (Training): ${np.sqrt(mean_squared_error(y_train, y_pred_train_lr)):,.2f}")
print(f"RMSE (Testing): ${np.sqrt(mean_squared_error(y_test, y_pred_test_lr)):,.2f}")
print(f"MAE (Testing): ${mean_absolute_error(y_test, y_pred_test_lr):,.2f}")

# 9. Model Building - Random Forest Regressor
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

y_pred_train_rf = rf_model.predict(X_train)
y_pred_test_rf = rf_model.predict(X_test)

print("=== RANDOM FOREST ===")
print(f"R² Score (Training): {r2_score(y_train, y_pred_train_rf):.4f}")
print(f"R² Score (Testing): {r2_score(y_test, y_pred_test_rf):.4f}")
print(f"RMSE (Training): ${np.sqrt(mean_squared_error(y_train, y_pred_train_rf)):,.2f}")
print(f"RMSE (Testing): ${np.sqrt(mean_squared_error(y_test, y_pred_test_rf)):,.2f}")
print(f"MAE (Testing): ${mean_absolute_error(y_test, y_pred_test_rf):,.2f}")

# 10. Feature Importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("Top 10 Most Important Features:")
print(feature_importance.head(10))

plt.figure(figsize=(10, 6))
plt.barh(
    feature_importance['Feature'].head(10),
    feature_importance['Importance'].head(10),
    color='steelblue'
)
plt.xlabel('Importance Score')
plt.title('Top 10 Most Important Features', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# 11. Model Comparison and Predictions
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(y_test, y_pred_test_lr, alpha=0.5, color='blue')
axes[0].plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--',
    lw=2
)
axes[0].set_title(
    'Linear Regression - Actual vs Predicted',
    fontsize=12,
    fontweight='bold'
)
axes[0].set_xlabel('Actual Price ($)')
axes[0].set_ylabel('Predicted Price ($)')
axes[0].grid(True, alpha=0.3)

axes[1].scatter(y_test, y_pred_test_rf, alpha=0.5, color='green')
axes[1].plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--',
    lw=2
)
axes[1].set_title(
    'Random Forest - Actual vs Predicted',
    fontsize=12,
    fontweight='bold'
)
axes[1].set_xlabel('Actual Price ($)')
axes[1].set_ylabel('Predicted Price ($)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nModel comparison completed!")
print("\nStudent: Rahul Rathod")
print("Email: rahulrathodwork13@gmail.com")

# 12. Insights and Conclusions
print("=" * 60)
print("SUMMARY OF FINDINGS")
print("=" * 60)

print("\nDATASET OVERVIEW:")
print(f"Total Houses Analyzed: {len(df):,}")
print(f"Average Price: ${df['price'].mean():,.2f}")
print(f"Average Bedrooms: {df['bedrooms'].mean():.2f}")
print(f"Average Living Area: {df['sqft_living'].mean():,.0f} sqft")

print("\nBEST PERFORMING MODEL: Random Forest")
print(f"R² Score: {r2_score(y_test, y_pred_test_rf):.4f}")
print(f"RMSE: ${np.sqrt(mean_squared_error(y_test, y_pred_test_rf)):,.2f}")
print(f"MAE: ${mean_absolute_error(y_test, y_pred_test_rf):,.2f}")

print("\nKEY FEATURES AFFECTING PRICE:")
for idx, row in feature_importance.head(5).iterrows():
    print(f"{idx + 1}. {row['Feature']}: {row['Importance']:.4f}")

print("\nPROJECT COMPLETED SUCCESSFULLY!")
print("=" * 60)
