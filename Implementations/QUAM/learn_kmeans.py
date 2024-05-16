import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import joblib

features = np.load("/ASR_Kmeans/QUAM/features.npy")

k_clusters = 50

kmeans_model = KMeans(n_clusters=k_clusters)

kmeans_model.fit(features)

joblib.dump(kmeans_model, "/ASR_Kmeans/QUAM/models/kmeans_model.sav")

centroids = kmeans_model.cluster_centers_

img_ids = []


for i in range(len(centroids)):
    min = np.inf
    min_id = 0
    for j in range(len(features)):
        if np.linalg.norm(centroids[i] - features[j]) < min:
            min = np.linalg.norm(centroids[i] - features[j])
            min_id = j

    img_ids.append(min_id)

print(img_ids)
