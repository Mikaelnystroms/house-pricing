#!/usr/bin/env python
# coding: utf-8

import os
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import itertools
import bentoml

# Load and extract data
data_dir = '../data'
for file in os.listdir(data_dir):
    if file.endswith('.zip'):
        zip_path = os.path.join(data_dir, file)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
            print(f'Extracted {file}')

# Read data
train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))

# Print dataset info
print("\nDataset Info")
print("\nTraining set shape:", train_df.shape)
print("Test set shape:", test_df.shape)
print("\nTraining set info:")
print(train_df.info())

# Print numerical features summary
print("\nNumerical Features Summary")
print(train_df.describe())

# Analyze missing values
print("\nMissing Values Analysis")
missing_train = train_df.isnull().sum().sort_values(ascending=False)
missing_train_pct = (missing_train / len(train_df)) * 100
missing_summary = pd.DataFrame({
    'Missing Count': missing_train,
    'Missing Percentage': missing_train_pct
})
print(missing_summary[missing_summary['Missing Count'] > 0])

# Handle missing values
print("\nMissing Values Strategy")
print("Based on the analysis, we'll handle missing values as follows:")

high_missing = missing_summary[missing_summary['Missing Percentage'] > 80].index
print("\nFeatures to consider dropping (>80% missing):")
print(high_missing.tolist())

numerical_features = train_df.select_dtypes(include=[np.number]).columns
categorical_features = train_df.select_dtypes(include=['object']).columns

print("\nImputation strategy for notable features:")
for feature in missing_summary[missing_summary['Missing Count'] > 0].index:
    if feature in numerical_features:
        print(f"- {feature}: Median imputation")
    elif feature in categorical_features:
        print(f"- {feature}: Mode imputation")

plt.figure(figsize=(10, 6))
sns.heatmap(train_df.isnull(), yticklabels=False, cbar=True, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()

# Drop high-missing features
train_df = train_df.drop(columns=['PoolQC', 'MiscFeature', 'Alley', 'Fence'])
print("\nDropped high-missing features")

# Impute numerical features
num_features = ['LotFrontage', 'GarageYrBlt', 'MasVnrArea']
for feat in num_features:
    train_df = train_df.assign(**{feat: train_df[feat].fillna(train_df[feat].median())})

# Impute categorical features
categorical_features = [
    'MasVnrType', 'FireplaceQu', 'GarageQual', 'GarageFinish',
    'GarageType', 'GarageCond', 'BsmtFinType2', 'BsmtExposure', 
    'BsmtCond', 'BsmtQual', 'BsmtFinType1', 'Electrical'
]
for feat in categorical_features:
    train_df = train_df.assign(**{feat: train_df[feat].fillna(train_df[feat].mode()[0])})
print("\nMode imputation complete")

print(f"\nMissing values remaining: {train_df.isnull().sum().sum()}")

# Analyze sale price distribution
plt.figure(figsize=(10, 6))
sns.histplot(train_df['SalePrice'], kde=True)
plt.title('Distribution of Sale Prices')
plt.xlabel('Sale Price')
plt.show()

print("\nSale Price Skewness:", stats.skew(train_df['SalePrice']))

plt.figure(figsize=(10, 6))
sns.histplot(np.log1p(train_df['SalePrice']), kde=True)
plt.title('Distribution of Log-Transformed Sale Prices')
plt.xlabel('Log Sale Price')
plt.show()

# Correlation analysis
numerical_features = train_df.select_dtypes(include=[np.number]).columns
correlation_matrix = train_df[numerical_features].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Numerical Features')
plt.show()

correlations = correlation_matrix['SalePrice'].sort_values(ascending=False)
print("\nTop 10 Correlations with Sale Price")
print(correlations[:10])

# Plot top correlations
top_corr_features = correlations[1:6].index
plt.figure(figsize=(15, 10))
for i, feature in enumerate(top_corr_features, 1):
    plt.subplot(2, 3, i)
    plt.scatter(train_df[feature], train_df['SalePrice'], alpha=0.5)
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(f'SalePrice vs {feature}')
plt.tight_layout()
plt.show()

# Analyze categorical features
categorical_features = train_df.select_dtypes(include=['object']).columns
print("\n=== Categorical Features ===")
for feature in categorical_features:
    print(f"\nValue counts for {feature}:")
    print(train_df[feature].value_counts().head())

# Plot important categorical features
important_categorical = ['OverallQual', 'Neighborhood', 'HouseStyle', 'SaleCondition']
plt.figure(figsize=(15, 10))
for i, feature in enumerate(important_categorical, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x=feature, y='SalePrice', data=train_df)
    plt.xticks(rotation=45)
    plt.title(f'Sale Price by {feature}')
plt.tight_layout()
plt.show()

# Plot year-related features
year_features = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']
plt.figure(figsize=(15, 5))
for i, feature in enumerate(year_features, 1):
    plt.subplot(1, 3, i)
    plt.scatter(train_df[feature], train_df['SalePrice'], alpha=0.5)
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(f'SalePrice vs {feature}')
plt.tight_layout()
plt.show()

# Prepare data for modeling
features = train_df.drop(['SalePrice', 'Id'], axis=1)
features = pd.get_dummies(features)
features = features.fillna(features.mean())

X = features
y = train_df['SalePrice']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Train and evaluate models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42)
}

print("\nModel Performance")
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_val_scaled)
    
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, y_pred)
    
    cv_scores = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for train_idx, val_idx in kf.split(X_train_scaled):
        X_cv_train = X_train_scaled[train_idx]
        y_cv_train = y_train.iloc[train_idx]
        X_cv_val = X_train_scaled[val_idx]
        y_cv_val = y_train.iloc[val_idx]
        
        model.fit(X_cv_train, y_cv_train)
        y_cv_pred = model.predict(X_cv_val)
        cv_scores.append(r2_score(y_cv_val, y_cv_pred))
    
    cv_scores = np.array(cv_scores)
    
    print(f"\n{name}:")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"R2 Score: {r2:.4f}")
    print(f"CV R2: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Plot feature importance
rf_model = models['Random Forest']
importance = pd.DataFrame({
    'feature': features.columns,
    'importance': rf_model.feature_importances_
}).nlargest(10, 'importance')

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=importance)
plt.title('Top 10 Features (Random Forest)')
plt.tight_layout()
plt.show()

# Hyperparameter tuning for XGBoost
param_grid = {
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5],
    'n_estimators': [100, 200],
    'min_child_weight': [1, 3],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 0.1]
}

kf = KFold(n_splits=3, shuffle=True, random_state=42)
best_score = float('inf')
best_params = None
total_iterations = np.prod([len(v) for v in param_grid.values()])
current_iteration = 0

# Grid search
for params in (dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())):
    cv_scores = []
    current_iteration += 1
    
    if current_iteration % 10 == 0:
        print(f"Progress: {current_iteration}/{total_iterations}")
    
    for train_idx, val_idx in kf.split(X_train_scaled):
        X_cv_train, X_cv_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model = XGBRegressor(**params, random_state=42)
        model.fit(X_cv_train, y_cv_train, eval_set=[(X_cv_val, y_cv_val)], verbose=False)
        
        y_cv_pred = model.predict(X_cv_val)
        cv_scores.append(np.sqrt(mean_squared_error(y_cv_val, y_cv_pred)))
    
    mean_rmse = np.mean(cv_scores)
    if mean_rmse < best_score:
        best_score = mean_rmse
        best_params = params
        print(f"\nNew best RMSE: ${mean_rmse:,.2f}")
        print(f"Parameters: {params}")

print(f"\nBest parameters: {best_params}")
print(f"Best RMSE: ${best_score:,.2f}")

# Train final model with best parameters
best_xgb = XGBRegressor(**best_params, random_state=42)
best_xgb.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False)

y_pred = best_xgb.predict(X_val_scaled)
final_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
final_r2 = r2_score(y_val, y_pred)

print("f\nFinal model metrics:")
print(f"RMSE: ${final_rmse:,.2f}")
print(f"R2: {final_r2:.4f}")

# Feature importance for best model
importance_df = pd.DataFrame({
    'feature': features.columns,
    'importance': best_xgb.feature_importances_
}).sort_values('importance', ascending=False).head(10)

print("\nTop 10 features:")
for _, row in importance_df.iterrows():
    print(f"{row['feature']}: {row['importance']:.4f}")

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=importance_df)
plt.title('Top 10 Features (Tuned XGBoost)')
plt.tight_layout()
plt.show()

# Create simplified model with most important features
top_features = [
    'OverallQual',
    'GrLivArea',
    'GarageCars',
    'GarageArea', 
    'TotalBsmtSF',
    'FullBath',
    'TotRmsAbvGrd',
    'YearBuilt'
]

# Prepare simplified data
X_train_simple = X_train[top_features]
X_val_simple = X_val[top_features]

# Create typical house profile
remaining_features = [col for col in X_train.columns if col not in top_features]
typical_house = pd.DataFrame(columns=X_train.columns)
typical_house.loc[0] = X_train.median()

# Handle categorical features
for col in remaining_features:
    if col.startswith(('MSZoning_', 'Street_', 'Alley_', 'LotShape_', 'LandContour_',
                      'Utilities_', 'LotConfig_', 'LandSlope_', 'Neighborhood_',
                      'Condition1_', 'Condition2_', 'BldgType_', 'HouseStyle_',
                      'RoofStyle_', 'RoofMatl_', 'Exterior1st_', 'Exterior2nd_',
                      'MasVnrType_', 'ExterQual_', 'ExterCond_', 'Foundation_',
                      'BsmtQual_', 'BsmtCond_', 'BsmtExposure_', 'BsmtFinType1_',
                      'BsmtFinType2_', 'Heating_', 'HeatingQC_', 'CentralAir_',
                      'Electrical_', 'KitchenQual_', 'Functional_', 'FireplaceQu_',
                      'GarageType_', 'GarageFinish_', 'GarageQual_', 'GarageCond_',
                      'PavedDrive_', 'SaleType_', 'SaleCondition_')):
        prefix = '_'.join(col.split('_')[:-1]) + '_'
        related_cols = [c for c in remaining_features if c.startswith(prefix)]
        typical_house.loc[0, related_cols] = 0
        most_common = X_train[related_cols].idxmax(axis=1).mode()[0]
        typical_house.loc[0, most_common] = 1

# Scale simplified features
scaler_simple = StandardScaler()
X_train_simple_scaled = scaler_simple.fit_transform(X_train_simple)
X_val_simple_scaled = scaler_simple.transform(X_val_simple)

# Train simplified model
simple_params = {
    'learning_rate': 0.1,
    'max_depth': 4,
    'n_estimators': 300,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0
}

simple_xgb = XGBRegressor(**simple_params, random_state=42)
simple_xgb.fit(
    X_train_simple_scaled,
    y_train,
    eval_set=[(X_val_simple_scaled, y_val)],
    verbose=False
)

# Evaluate simplified model
y_pred_simple = simple_xgb.predict(X_val_simple_scaled)
simple_rmse = np.sqrt(mean_squared_error(y_val, y_pred_simple))
simple_r2 = r2_score(y_val, y_pred_simple)

print("Simplified model performance:")
print(f"RMSE: ${simple_rmse:,.2f}")
print(f"R2 Score: {simple_r2:.4f}")

print("\nModel comparison:")
print(f"Full RMSE: ${final_rmse:,.2f}")
print(f"Simple RMSE: ${simple_rmse:,.2f}")
print(f"Full R2: {final_r2:.4f}")
print(f"Simple R2: {simple_r2:.4f}")

# Save models
bentoml.xgboost.save_model('house-pricing-model-simple', simple_xgb)
bentoml.sklearn.save_model('house-price-scaler-simple', scaler_simple)
bentoml.xgboost.save_model('house-pricing-model', best_xgb)

# Print features
print("Features")
for feature in features.columns:
    print(f"- {feature}")

booster = bentoml.xgboost.load_model("house-pricing-model-simple:latest")

test_data = np.array([[
    8,        # OverallQual
    2000,     # GrLivArea  
    2,        # GarageCars
    400,      # GarageArea
    1000,     # TotalBsmtSF
    2,        # FullBath
    8,        # TotRmsAbvGrd
    1960      # YearBuilt
]]).reshape(1, -1)

# Make prediction
res = booster.predict(test_data)
print(f" ${res[0]:,.2f}")
