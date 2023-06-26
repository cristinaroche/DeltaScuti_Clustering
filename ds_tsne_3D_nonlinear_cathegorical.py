# 3D plots and T-SNE for non-linear cathegorical features

from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

stars_dataset_new_cathegorical = pd.read_csv('stars_dataset_new_cathegorical.csv')
stars_dataset_nonlinear = stars_dataset_new_cathegorical[['armonics_first_cat', 'comb_suma_cat', 'comb_resta_cat', 'multiperiodic']].copy()
stars_dataset_nonlinear_type = stars_dataset_new_cathegorical[['type', 'armonics_first_cat', 'comb_suma_cat', 'comb_resta_cat', 'multiperiodic']].copy()
stars_dataset_nonlinear_amplitude = stars_dataset_new_cathegorical[['type', 'armonics_first_cat', 'comb_suma_cat', 'comb_resta_cat', 'f1_amp_sqrt']].copy()
# Compute the t-SNE transformation with 3 components
tsne = TSNE(n_components=3)
tsne_results = tsne.fit_transform(stars_dataset_nonlinear)

# Create a new DataFrame with the t-SNE results
df_tsne = pd.DataFrame(data = tsne_results, columns = ['Component 1', 'Component 2', 'Component 3'])

# Create the 3D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df_tsne['Component 1'], df_tsne['Component 2'], df_tsne['Component 3'])

ax.set_xlabel('Component 1')
ax.set_ylabel('Component 2')
ax.set_zlabel('Component 3')




# Create the 3D plot
# Define a function to add jitter
def jitter(arr):
    stdev = .03*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter(jitter(stars_dataset_nonlinear['armonics_first_cat']), jitter(stars_dataset_nonlinear['comb_suma_cat']), jitter(stars_dataset_nonlinear['comb_resta_cat']), c=stars_dataset_nonlinear['multiperiodic'], cmap='viridis')

ax.set_xlabel('armonics_first_cat')
ax.set_ylabel('comb_suma_cat')
ax.set_zlabel('comb_resta_cat')
fig.colorbar(p)


# Create the 3D plot
# Define a function to add jitter
def jitter(arr):
    stdev = .03*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev

colors = ['#D55E00', '#0072B2']  # specify the colors you want, can be any number of colors
cmap = LinearSegmentedColormap.from_list("my_colormap", colors)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter(jitter(stars_dataset_nonlinear_type['armonics_first_cat']), jitter(stars_dataset_nonlinear_type['comb_suma_cat']), jitter(stars_dataset_nonlinear_type['comb_resta_cat']), c=stars_dataset_nonlinear_type['type'], cmap=cmap)

ax.set_xlabel('armonics_first_cat')
ax.set_ylabel('comb_suma_cat')
ax.set_zlabel('comb_resta_cat')
fig.colorbar(p)


# Create the 3D plot
# Define a function to add jitter
def jitter(arr):
    stdev = .03*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev

colors = ['#0072B2', '#D55E00']  # specify the colors you want, can be any number of colors
cmap = LinearSegmentedColormap.from_list("my_colormap", colors)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter(jitter(stars_dataset_nonlinear_amplitude['armonics_first_cat']), jitter(stars_dataset_nonlinear_amplitude['comb_suma_cat']), jitter(stars_dataset_nonlinear_amplitude['comb_resta_cat']), c=stars_dataset_nonlinear_amplitude['f1_amp_sqrt'], cmap=cmap)

ax.set_xlabel('armonics_first_cat')
ax.set_ylabel('comb_suma_cat')
ax.set_zlabel('comb_resta_cat')
fig.colorbar(p)




plt.show()
