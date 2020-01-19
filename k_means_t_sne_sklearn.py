import pandas as pd
import os
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

directory = 'C:/AGA_studia/inzynierka/DATA/ivectory_centr'
IVectors = dict()
for file in os.listdir(directory):
    f = open(directory+"/"+file)
    ivector = f.read().split()
    for el in ivector:
        index = ivector.index(el)
        el = float(el)
        ivector[index] = el
    IVectors[file.split(".")[0]] = ivector
X = list(IVectors.values())
tsne = TSNE(n_components=2, random_state=0)
X_embedded = tsne.fit_transform(list(IVectors.values()))
df = pd.DataFrame({
    'x': X_embedded[:, 0],
    'y': X_embedded[:, 1]
})

kmeans = KMeans(n_clusters=4)
kmeans.fit(df)
labels = kmeans.predict(df)
centroids = kmeans.cluster_centers_
Centers = {}
i = 0
for key in (list(IVectors.keys())):
    Centers[key] = labels[i]
    i += 1
colmap = {0: 'r', 1: 'g', 2: 'b', 3: 'y'}
colmap_ = {"drzwi": 'r', "muzyka": 'g', "swiatlo": 'b', "temp": 'y'}
for i in range(0, len(X_embedded[:, 0])):
    plt.scatter(X_embedded[i, 0], X_embedded[i, 1], color=colmap_[list(IVectors.keys())[i].split("_")[-2]], s=5)
for i in range(len(centroids)):
    plt.scatter(*centroids[i], color=colmap[i], edgecolors="black", s=20)
plt.title("T-SNE")
plt.show()
print()
temp = [0, 0]
drzwi = [0, 0]
idx = 0
for value in list(Centers.values()):
    if list(Centers.keys())[idx].split("_")[-2] == "drzwi":
        drzwi[value] = drzwi[value] + 1
    if list(Centers.keys())[idx].split("_")[-2] == "temp":
        temp[value] = temp[value] + 1
    idx += 1