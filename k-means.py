import pandas as pd
import os
from sklearn.cluster import KMeans


directory = 'C:/AGA_studia/inzynierka/DATA/ivectory_sklejone_zdania_po_5_scentralizowane_grupami'
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


kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
labels = kmeans.predict(X)
centroids = kmeans.cluster_centers_
Centers = {}
i = 0
for key in (list(IVectors.keys())):
    Centers[key] = labels[i]
    i += 1
print()
temp = []
drzwi = []
swiatlo = []
muz = []
for i in range(4):
    temp.append(0)
    drzwi.append(0)
    muz.append(0)
    swiatlo.append(0)

idx = 0
for value in list(Centers.values()):
    if list(Centers.keys())[idx].split("_")[-2] == "drzwi":
        drzwi[value] = drzwi[value] + 1
    if list(Centers.keys())[idx].split("_")[-2] == "temp":
        temp[value] = temp[value] + 1
    if list(Centers.keys())[idx].split("_")[-2] == "swiatlo":
        swiatlo[value] = swiatlo[value] + 1
    if list(Centers.keys())[idx].split("_")[-2] == "muzyka":
        muz[value] = muz[value] + 1
    idx += 1

print(drzwi)
print(muz)
print(swiatlo)
print(temp)
print()
