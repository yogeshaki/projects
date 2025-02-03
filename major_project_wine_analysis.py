# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LassoCV, Ridge
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import shap
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load and Preprocess the Data
# Load dataset (replace with your actual path)
data = pd.read_csv(r'C:\Users\yogesh\Downloads\upload_b17b4ebe-b5ef-4168-b23e-759428717a8b.csv')

# Apply KNN Imputer for missing values
imputer = KNNImputer(n_neighbors=3)
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Split into input features and target
X = data_imputed.drop('quality', axis=1)
y = data_imputed['quality']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 2: Outlier Detection using Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
outliers = iso_forest.fit_predict(X_train_scaled)

# Visualize outliers
sns.scatterplot(x=X_train_scaled[:, 0], y=X_train_scaled[:, 1], hue=outliers, palette='coolwarm')
plt.title('Outlier Detection with Isolation Forest')
plt.show()

# Step 3: Feature Selection using Lasso Regression
lasso = LassoCV(cv=5)
lasso.fit(X_train_scaled, y_train)

# Get important features
importance = np.abs(lasso.coef_)
feature_names = X.columns
important_features = pd.Series(importance, index=feature_names).sort_values(ascending=False)

# Plot feature importance
important_features.plot(kind='bar', title='Feature Importance from Lasso')
plt.show()

# --- Discretize Target Variable ---
# Discretize quality scores into categorical values
# For example: 0-4 = Poor (0), 5-6 = Average (1), 7-10 = Excellent (2)
bins = [0, 4, 6, 10]  # Define bins
labels = [0, 1, 2]    # Define labels for the bins

# Apply binning to y_train and y_test
y_train_binned = pd.cut(y_train, bins=bins, labels=labels)
y_test_binned = pd.cut(y_test, bins=bins, labels=labels)

# Step 4: Classification using Random Forest with balanced class weights
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_classifier.fit(X_train_scaled, y_train_binned)

# Predictions using Random Forest (classification)
y_pred = rf_classifier.predict(X_test_scaled)

# Evaluate classification with the binned target, handle zero_division warnings
print(f'Random Forest Accuracy: {accuracy_score(y_test_binned, y_pred)}')
print(classification_report(y_test_binned, y_pred, zero_division=1))

# Step 5: Regression using Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)

# Predict on test data for regression
y_pred_ridge = ridge.predict(X_test_scaled)

# Evaluate regression model (continuous values)
print(f'R2 Score (Ridge): {r2_score(y_test, y_pred_ridge)}')
print(f'MSE (Ridge): {mean_squared_error(y_test, y_pred_ridge)}')

# Step 6: SHAP Explainer without additivity check in method call
explainer = shap.TreeExplainer(rf_classifier)
shap_values = explainer.shap_values(X_test_scaled, check_additivity=False)

# Visualize SHAP summary plot
shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_names)
