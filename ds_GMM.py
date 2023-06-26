import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.manifold import TSNE


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


# Define the model
gmm = GaussianMixture(
    n_components=3,  # Number of Gaussian distributions to fit to the data
    covariance_type='full',  # 'full' (each component has its own general covariance matrix), 'tied' (all components share the same general covariance matrix), 'diag' (each component has its own diagonal covariance matrix), 'spherical' (each component has its own single variance)
    tol=1e-3,  # Convergence threshold
    reg_covar=1e-6,  # Regularization
    max_iter=100,  # Maximum number of iterations
    n_init=1  # Number of initializations to perform
)

# Fit the model
gmm.fit(df_scaled)

# Predict the clusters with the .predict method of the object
labels = gmm.predict(df_scaled)

# Apply PCA
# Compute the t-SNE transformation with 3 components
tsne = TSNE(n_components=3)
tsne_results = tsne.fit_transform(df_scaled)
# Create a new DataFrame with the t-SNE results
df_tsne = pd.DataFrame(data = tsne_results, columns = ['Component 1', 'Component 2', 'Component 3'])
# Create the 3D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df_tsne['Component 1'], df_tsne['Component 2'], df_tsne['Component 3'], c=labels, cmap='viridis')
ax.set_xlabel('Component 1')
ax.set_ylabel('Component 2')
ax.set_zlabel('Component 3')

# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(df_tsne['Component 1'], df_tsne['Component 2'], df_tsne['Component 3'], c=s_ds['type'], cmap='viridis')
# ax.set_xlabel('Component 1')
# ax.set_ylabel('Component 2')
# ax.set_zlabel('Component 3')


plt.show()