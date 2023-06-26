# This script applies the feature selection technique from wrapper-based methods called "Recursive Feature Elimination (RFE)"

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


##### load the data into a dataframe
s_ds = pd.read_csv('stars_dataset_new.csv')

# Select your features
# Change type from 1 and 2 to HADS and LADS to appear correctly in the legend
for i in range(len(s_ds)):
    if s_ds['type'][i] == 'HADS':
        s_ds['type'][i] = 1
    elif s_ds['type'][i] == 'LADS':
        s_ds['type'][i] = 2

features = ['type','f1_freq', 'f1_amp_sqrt', 'f1_fase', 'f2_freq', 'f2_amp_sqrt', 'arm_first', 'arm_sec', 'comb_suma', 'comb_rest']
df_features = s_ds[features]

# Scale the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_features)

# Generate a synthetic dataset
X = df_scaled['f1_freq', 'f1_amp_sqrt', 'f1_fase', 'f2_freq', 'f2_amp_sqrt', 'arm_first', 'arm_sec', 'comb_suma', 'comb_rest']
y = df_scaled['type']

# Convert DataFrames to numpy arrays for classifier to work
X = X.values
y = y.values

# Create a base model for feature selection
# model = LogisticRegression()
model = RandomForestClassifier() # More robust to outliers and skewed data

# Create the RFE object and specify the number of desired features to select
rfe = RFE(estimator=model, n_features_to_select=5)

# Fit the RFE object to the data
rfe.fit(X, y)

# Get the selected features
selected_features = rfe.support_
selected_feature_names = [name for name, selected in zip(X.columns, selected_features) if selected]

print("Selected Features:")
print(selected_feature_names)

# Get the feature importance rankings
feature_importance = rfe.estimator_.feature_importances_

# Sort the feature names and importances in descending order
sorted_indices = np.argsort(feature_importance)[::-1]
sorted_feature_names = np.array(X.columns)[sorted_indices]
sorted_feature_importance = feature_importance[sorted_indices]

# Plotting the feature importance rankings
plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_feature_names)), sorted_feature_importance, align='center')
plt.yticks(range(len(sorted_feature_names)), sorted_feature_names)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importance Rankings')
plt.show()












