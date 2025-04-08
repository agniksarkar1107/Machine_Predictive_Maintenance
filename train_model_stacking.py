import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load the data
print("Loading and preprocessing data...")
data = pd.read_csv('predictive_maintenance.csv')

# Data preprocessing
# Drop unnecessary columns
data = data.drop(["UDI", "Product ID"], axis=1)

# Convert temperature from Kelvin to Celsius
data["Air temperature [°C]"] = data["Air temperature [K]"] - 273.15
data["Process temperature [°C]"] = data["Process temperature [K]"] - 273.15
        
# Create a new column for the temperature difference
data["Temperature difference [°C]"] = data["Process temperature [°C]"] - data["Air temperature [°C]"]
        
# Drop original temperature columns in Kelvin
data = data.drop(columns=["Air temperature [K]", "Process temperature [K]"])

# Clean column names by removing special characters
data.columns = [col.replace('[', '_').replace(']', '_').replace('°', '').replace(' ', '_') for col in data.columns]

# Encode categorical variables
le = LabelEncoder()
data['Type'] = le.fit_transform(data['Type'])
data['Failure_Type'] = le.fit_transform(data['Failure_Type'])

# Split features and target
X = data.drop(['Target', 'Failure_Type'], axis=1)
y = data['Target']
y_failure_type = data['Failure_Type']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)

print("Data preprocessing complete.")
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Define best parameters for XGBoost (based on your reference)
print("Training XGBoost model...")
xgb_best = xgb.XGBClassifier(
    colsample_bytree=1.0, 
    learning_rate=0.3, 
    max_depth=3, 
    n_estimators=50, 
    subsample=0.8,
    random_state=42
)
xgb_best.fit(X_train, y_train)
print("XGBoost model training complete.")

# Define best parameters for Random Forest (based on your reference)
print("Training Random Forest model...")
rf_best = RandomForestClassifier(
    max_depth=20, 
    min_samples_leaf=1, 
    min_samples_split=2, 
    n_estimators=50,
    random_state=42
)
rf_best.fit(X_train, y_train)
print("Random Forest model training complete.")

# Create the stacking model
print("Training stacking ensemble model...")
stacking_model = StackingClassifier(
    estimators=[('xgb', xgb_best), ('rf', rf_best)],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5
)
stacking_model.fit(X_train, y_train)
print("Stacking model training complete.")

# Train failure type classifier 
print("Training failure type classifier...")
failure_type_model = RandomForestClassifier(
    max_depth=20, 
    min_samples_leaf=1, 
    min_samples_split=2, 
    n_estimators=100,
    random_state=42
)
failure_type_model.fit(X_scaled_df, y_failure_type)
print("Failure type classifier training complete.")

# Evaluate base models
print("\nModel Evaluation:")
for name, model in [('XGBoost', xgb_best), ('Random Forest', rf_best), ('Stacking Ensemble', stacking_model)]:
    y_pred = model.predict(X_test)
    train_acc = model.score(X_train, y_train)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"{name}:")
    print(f"  Training Accuracy: {train_acc*100:.2f}%")
    print(f"  Test Accuracy: {test_acc*100:.2f}%")
    print(f"  Classification Report:\n{classification_report(y_test, y_pred)}")
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'confusion_matrix_{name.replace(" ", "_").lower()}.png')
    plt.close()

# Get feature importance from Random Forest
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_best.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print("\nTop 5 Most Important Features:")
print(feature_importance.head())

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance')
plt.savefig('feature_importance.png')
plt.close()

# Create a wrapper model for the stacked model
class StackedModel:
    def __init__(self, stacking_model, scaler, failure_type_model):
        self.stacking_model = stacking_model
        self.scaler = scaler
        self.failure_type_model = failure_type_model
        self.feature_names = X.columns.tolist()
        
    def predict(self, X):
        # Scale the input
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        return self.stacking_model.predict(X_scaled)
    
    def predict_proba(self, X):
        # Scale the input
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        return self.stacking_model.predict_proba(X_scaled)
    
    def predict_failure_type(self, X):
        # Scale the input
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        return self.failure_type_model.predict(X_scaled)

# Create the wrapper model
final_model = StackedModel(stacking_model, scaler, failure_type_model)

# Save models and preprocessing objects
print("\nSaving models...")
joblib.dump(final_model, 'stacked_model.joblib')
joblib.dump(le, 'label_encoder.joblib')
print("Models saved successfully!")

print("\nTraining and evaluation complete!") 