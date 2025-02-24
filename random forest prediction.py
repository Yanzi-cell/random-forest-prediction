import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split  # Import this
import pandas as pd
import matplotlib.pyplot as plt

# Path to your training Excel file
training_file_path = r"C:\Users\Lenovo\Desktop\prediction 1.xlsx"

# Define column names
column_names = ["category", "bacteria", "f1", "f2", "f3", "f4"]

# Load training data
df = pd.read_excel(training_file_path, sheet_name='RAW4', header=None, names=column_names, skiprows=0)

features = df[['f1', 'f2', 'f3', 'f4']]
categories = df['category']  # Assuming 'category' column contains your categories
# Handle missing values
imputer = SimpleImputer(strategy='mean')
features = pd.DataFrame(imputer.fit_transform(features), columns=features.columns)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, categories, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict the categories on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Data Accuracy: {accuracy:.2f}")

# If you have a new dataset and want to predict categories
prediction_file_path = r"C:\Users\Lenovo\Desktop\prediction 1.xlsx"
prediction_df = pd.read_excel(prediction_file_path, sheet_name='Prediction4', header=None, names=column_names, skiprows=0)

# Define column names
column_names = ["category", "bacteria", "f1", "f2", "f3", "f4"]
prediction_features = prediction_df[['f1', 'f2', 'f3', 'f4']]

# Handle missing values in the prediction data
prediction_features = pd.DataFrame(imputer.transform(prediction_features), columns=prediction_features.columns)

# Predict the categories for the new dataset
predicted_categories = model.predict(prediction_features)
prediction_df['Predicted Category'] = predicted_categories

# Output the predictions
print(prediction_df[['bacteria', 'Predicted Category']])