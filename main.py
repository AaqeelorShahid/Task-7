import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset 
dataset = pd.read_csv('UK.csv')
X = dataset.iloc[:, [1, 2]].values

# Using Elbow method to find number of Clusters 
from sklearn.cluster import KMeans

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    print(i)
    
plt.figure(figsize=(15, 10))
plt.plot (range(1, 11), wcss)
plt.title("The Elbow Method")
plt.xlabel("Number of cluster")
plt.ylabel("WCSS")
plt.show()

# Applying K-Means to dataset
kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, max_iter=300, random_state=0)
y_kmeans = kmeans.fit_predict(X)

# Visulizing the data in Scatterplot
plt.figure(figsize=(15, 10))
plt.scatter (X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'blue', alpha=0.2, label = 'Cluster 1')
plt.scatter (X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'purple', alpha=0.5, label = 'Cluster 2')

plt.scatter (kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], alpha=0.5, s=300, c = 'cyan', label='Centroids')

plt.title("Clusters of CO2 Emission (UK)")
plt.xlabel("Year", size = 14)
plt.ylabel("Average CO2 Emission tones", size = 14)
plt.legend()
plt.show()
