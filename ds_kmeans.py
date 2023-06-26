# It applies k-means clustering technique.
# Three times using different set of features and compares between them the labels and coincidence in results

import matplotlib.pyplot as plt  # data visualization library
from kneed import KneeLocator
from sklearn.cluster import KMeans # machine learning library
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import glob
import os 
import csv
import pickle
                 
# Hyperparameters of k-means
num_clusters = 2
num_init = 10
maxi_iter = 300
seed = 42

# load the data into a dataframe
stars_dataset = pd.read_csv('stars_dataset.csv')
# Cargamos el diccionario desde el archivo externo
with open('stars_dict.pickle', 'rb') as dict_stars_data_processed:
    stars_dict_cargado = pickle.load(dict_stars_data_processed)

# List of variable names to extract
var_list = ['f1_freq', 'f1_amp', 'f1_fase', 'arm_first', 'comb_suma', 'comb_rest']

# Create an empty array to store the data
data_array = np.zeros((len(stars_dict_cargado), len(var_list)))

# Loop through each case in the dictionary
for i, case in enumerate(stars_dict_cargado): 
    # Loop through each variable in var_list
    for j, var in enumerate(var_list):
        # save cases in rows and variables of each case in their respective columns
        data_array[i,j] = stars_dict_cargado[case][var]

data_array_scaled = np.zeros((len(stars_dict_cargado), len(var_list)))
scaler = StandardScaler()
data_array_scaled = scaler.fit_transform(data_array) #  standardize the data by scaling each feature to have a mean of 0 and a variance of 1

kmeans = KMeans(init="random", n_clusters=num_clusters, n_init=num_init, max_iter=maxi_iter, random_state=seed) #Create a KMeans object
#n_init=10 number of times the algorithm will be run with different initializations, with the best results being kept.
#max_iter=300 maximum number of iterations the algorithm will perform before stopping
#random_state=42 sets a seed for the random number generator, which ensures that the same results are obtained each time the code is run
kmeans.fit(data_array_scaled) # fit method fits the model to the stars data, treats each row of the scaled_features array as a separate data point with the same number of features, and tries to cluster them based on their similarity in feature space.

# The lowest SSE value
kmeans.inertia_ #alculates the sum of squared distances of data points to their nearest cluster center, which is a measure of how well the data points are clustered around their respective centers
# Final locations of the centroid 
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
# The number of iterations required to converge
kmeans.n_iter_
kmeans.labels_[:5] #returns the predicted cluster labels for the first five data points. The labels are based on the cluster centers obtained by the algorithm after convergence.

print('Inertia is ', kmeans.inertia_, 'with number of clusters', num_clusters)
print('Iteration of convergence' , kmeans.n_iter_)
print('Stars that belong to group A: ', np.count_nonzero(kmeans.labels_==0))
print('Stars that belong to group B: ', np.count_nonzero(kmeans.labels_==1))
print('Stars that belong to group C: ', np.count_nonzero(kmeans.labels_==2))
print('Stars that belong to group D: ', np.count_nonzero(kmeans.labels_==3))
print('Stars that belong to group E: ', np.count_nonzero(kmeans.labels_==4))
print('Cluster centers', cluster_centers)

##
# Load the dictionary from the external file
with open('stars_dict.pickle', 'rb') as dict_stars_data_processed:
    stars_dict_cargado = pickle.load(dict_stars_data_processed)

# List of variable names to extract
#var_list2 = ['f1_freq', 'f1_amp', 'f2_amp', 'arm_first', 'arm_sec', 'comb_suma', 'comb_rest']
var_list2 = ['f1_freq', 'f1_amp', 'arm_first', 'arm_sec', 'comb_suma', 'comb_rest']

# Create an empty array to store the data
data_array2 = np.zeros((len(stars_dict_cargado), len(var_list2)))

# Loop through each case in the dictionary
for i, case in enumerate(stars_dict_cargado): 
    # Loop through each variable in var_list
    for j, var in enumerate(var_list2):
        # save cases in rows and variables of each case in their respective columns
        data_array2[i,j] = stars_dict_cargado[case][var]

data_array_scaled2 = np.zeros((len(stars_dict_cargado), len(var_list2)))
scaler = StandardScaler()
data_array_scaled2 = scaler.fit_transform(data_array2) #  standardize the data by scaling each feature to have a mean of 0 and a variance of 1

kmeans2 = KMeans(init="random", n_clusters=num_clusters, n_init=num_init, max_iter=maxi_iter, random_state=seed) #Create a KMeans object
#n_init=10 number of times the algorithm will be run with different initializations, with the best results being kept.
#max_iter=300 maximum number of iterations the algorithm will perform before stopping
#random_state=42 sets a seed for the random number generator, which ensures that the same results are obtained each time the code is run
kmeans2.fit(data_array_scaled2) # fit method fits the model to the stars data, treats each row of the scaled_features array as a separate data point with the same number of features, and tries to cluster them based on their similarity in feature space.

# The lowest SSE value
kmeans2.inertia_ #alculates the sum of squared distances of data points to their nearest cluster center, which is a measure of how well the data points are clustered around their respective centers
# Final locations of the centroid 
cluster_centers1 = scaler.inverse_transform(kmeans2.cluster_centers_)
# The number of iterations required to converge
kmeans2.n_iter_
kmeans2.labels_[:5] #returns the predicted cluster labels for the first five data points. The labels are based on the cluster centers obtained by the algorithm after convergence.

print('Inertia is ', kmeans2.inertia_, 'with number of clusters', num_clusters)
print('Iteration of convergence' , kmeans2.n_iter_)
print('Stars that belong to group A: ', np.count_nonzero(kmeans2.labels_==0))
print('Stars that belong to group B: ', np.count_nonzero(kmeans2.labels_==1))
print('Stars that belong to group C: ', np.count_nonzero(kmeans2.labels_==2))
print('Stars that belong to group D: ', np.count_nonzero(kmeans2.labels_==3))
print('Stars that belong to group E: ', np.count_nonzero(kmeans2.labels_==4))
print('Cluster centers', cluster_centers1)

equality = kmeans.labels_ == kmeans2.labels_
print(equality)
np.sum(equality==True)
fail_cases = len(stars_dict_cargado)-np.sum(equality==True)
percentaje_coincident = np.sum(equality==True)/len(stars_dict_cargado)*100
print('They fail in', fail_cases, 'which represents', percentaje_coincident, '% of agreement')

#y_kmeans = kmeans.predict(scaler.fit_transform(X)) #predict the cluster labels for each data point in X. The resulting labels are stored in the variable y_kmeans





## No amplitude 1
# Cargamos el diccionario desde el archivo externo
with open('stars_dict.pickle', 'rb') as dict_stars_data_processed:
    stars_dict_cargado = pickle.load(dict_stars_data_processed)

# List of variable names to extract
#var_list3 = ['f1_freq', 'f2_amp', 'arm_first', 'arm_sec', 'comb_suma', 'comb_rest']
var_list3 = ['f1_freq', 'arm_first', 'comb_suma', 'comb_rest']
#var_list3 = ['f1_amp']

# Create an empty array to store the data
data_array3 = np.zeros((len(stars_dict_cargado), len(var_list3)))

# Loop through each case in the dictionary
for i, case in enumerate(stars_dict_cargado): 
    # Loop through each variable in var_list
    for j, var in enumerate(var_list3):
        # save cases in rows and variables of each case in their respective columns
        data_array3[i,j] = stars_dict_cargado[case][var]

data_array_scaled3 = np.zeros((len(stars_dict_cargado), len(var_list3)))
scaler = StandardScaler()
data_array_scaled3 = scaler.fit_transform(data_array3) #  standardize the data by scaling each feature to have a mean of 0 and a variance of 1

kmeans3 = KMeans(init="random", n_clusters=num_clusters, n_init=num_init, max_iter=maxi_iter, random_state=seed) #Create a KMeans object
#n_init=10 number of times the algorithm will be run with different initializations, with the best results being kept.
#max_iter=300 maximum number of iterations the algorithm will perform before stopping
#random_state=42 sets a seed for the random number generator, which ensures that the same results are obtained each time the code is run
kmeans3.fit(data_array_scaled3) # fit method fits the model to the stars data, treats each row of the scaled_features array as a separate data point with the same number of features, and tries to cluster them based on their similarity in feature space.

# The lowest SSE value
kmeans3.inertia_ #alculates the sum of squared distances of data points to their nearest cluster center, which is a measure of how well the data points are clustered around their respective centers
# Final locations of the centroid 
cluster_centers2 = scaler.inverse_transform(kmeans3.cluster_centers_)
# The number of iterations required to converge
kmeans3.n_iter_
kmeans3.labels_[:5] #returns the predicted cluster labels for the first five data points. The labels are based on the cluster centers obtained by the algorithm after convergence.

print('Inertia is ', kmeans3.inertia_, 'with number of clusters', num_clusters)
print('Iteration of convergence' , kmeans3.n_iter_)
print('Stars that belong to group A: ', np.count_nonzero(kmeans3.labels_==0))
print('Stars that belong to group B: ', np.count_nonzero(kmeans3.labels_==1))
print('Stars that belong to group C: ', np.count_nonzero(kmeans3.labels_==2))
print('Stars that belong to group D: ', np.count_nonzero(kmeans3.labels_==3))
print('Stars that belong to group E: ', np.count_nonzero(kmeans3.labels_==4))
print('Cluster centers', cluster_centers2)

equality = kmeans.labels_ == kmeans3.labels_
print(equality)
np.sum(equality==True)
fail_cases = len(stars_dict_cargado)-np.sum(equality==True)
percentaje_coincident = np.sum(equality==True)/len(stars_dict_cargado)*100
print('They fail in', fail_cases, 'which represents', percentaje_coincident, '% of agreement')
#for i,case in stars_dict_cargado:
#    good = stars_dict_cargado[case]['type'] == kmeans3.labels_[i]
idx = 0 
good= []
buenos = []
type_list=[]
labels3 = kmeans3.labels_.tolist()
labels3 = [x + 1 for x in labels3]
for case in stars_dict_cargado.keys():
    good.append(stars_dict_cargado[case]['type'] == labels3[idx])
    type_list.append(stars_dict_cargado[case]['type'])
    idx = idx+1
    n_buenos = np.count_nonzero(good)
    ñe = len(np.array(type_list)) - n_buenos
np.array(type_list)
for case, values in stars_dict_cargado.items():
    i = list(stars_dict_cargado.keys()).index(case)
    good2 = values['type'] == kmeans3.labels_[i]




labels3 = kmeans3.labels_.tolist()
labels3 = [x + 1 for x in labels3]
# get the values from stars_dict_cargado
values = [stars_dict_cargado[case]['type'] for case in stars_dict_cargado]
# initialize the coincide list with zeros
coincide = [0] * len(labels3)
for i in range(len(labels3)):
    coincide[i] = values[i] == labels3[i]

# create the color map
colors = { 0: 'green', 1: 'blue'}
# create the bar graph
x = np.arange(len(values))
width = 0.35
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, values, width, color=[colors[x] for x in values])
#rects2 = ax.bar(x + width/2, labels3, width, color=[colors[x] for x in labels3])


# add some labels and titles
ax.set_ylabel('Value')
ax.set_title('Comparison of Two Algorithms')
ax.legend((rects1[0], rects2[0]), ('Algorithm 1', 'Algorithm 2'))

plt.show()





clusters = [1,2,3,4,5,6,7]
inertia = [720,549.76,430.13,358.11,301,230,199]
inertia2 = [840,563.66,446.83,374.8,304,250,213]
inertia3 = [720,500.11,392.85,319.4,246,219,204]
""" fig, ax = plt.subplots(3)
ax[0].plot(clusters, inertia, '-o')
ax[0].set_title('#0')
ax[1].plot(clusters, inertia2, '-o')
ax[1].set_title('#1')
ax[2].plot(clusters, inertia3, '-o')
ax[2].set_title('#2')
plt.subplots_adjust(hspace=0.5) """
plt.plot(clusters, inertia, '-o', label='#0')
plt.plot(clusters, inertia2, '-o', label='#1')
plt.plot(clusters, inertia3, '-o', label='#2')
plt.legend()
plt.show()