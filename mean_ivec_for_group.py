from sklearn.cluster import KMeans
import os
import numpy as np

directory = 'C:/AGA_studia/inzynierka/DATA/ivectory_20x_to_samo'
list_drzwi = []
list_muz = []
list_swiatlo = []
list_temp = []
IVectors = dict()
for file in os.listdir(directory):
    f = open(directory + "/" + file)
    ivector = f.read().split()
    for el in ivector:
        index = ivector.index(el)
        el = float(el)
        ivector[index] = el
    if file.split("_")[-2] == "drzwi":
        list_drzwi.append(ivector)
    elif file.split("_")[-2] == "muzyka":
        list_muz.append(ivector)
    elif file.split("_")[-2] == "swiatlo":
        list_swiatlo.append(ivector)
    elif file.split("_")[-2] == "temp":
        list_temp.append(ivector)
    IVectors[file.split(".")[0]] = ivector

mean_muz = np.mean(list_muz, axis=0)
mean_drzwi = np.mean(list_drzwi, axis=0)
mean_swiatlo = np.mean(list_swiatlo, axis=0)
mean_temp = np.mean(list_temp, axis=0)

#centralizacja grupami
list_drzwi = {}
list_muz = {}
list_swiatlo = {}
list_temp = {}
for key in list(IVectors.keys()):
    if key.split("_")[-2] == "drzwi":
        IVectors[key] = np.array(IVectors[key]) - mean_drzwi
        list_drzwi[key] = (IVectors[key])
    if key.split("_")[-2] == "muzyka":
        IVectors[key] = np.array(IVectors[key]) - mean_muz
        list_muz[key] = (IVectors[key])
    if key.split("_")[-2] == "swiatlo":
        IVectors[key] = np.array(IVectors[key]) - mean_swiatlo
        list_swiatlo[key] = (IVectors[key])
    if key.split("_")[-2] == "temp":
        IVectors[key] = np.array(IVectors[key]) - mean_temp
        list_temp[key] = (IVectors[key])
'''
kmeans = KMeans(n_clusters=64)
kmeans.fit(list(list_drzwi.values()))
labels = kmeans.predict(list(list_drzwi.values()))
centroids = kmeans.cluster_centers_
Centers = {}
i = 0
for key in list(list_drzwi.keys()):
    Centers[key] = labels[i]
    i += 1
'''
X = list(list_drzwi.values())
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)
X_embedded = pca.fit_transform(X)
kmeans = KMeans(n_clusters=64)
kmeans.fit(X_embedded)
labels = kmeans.predict(X_embedded)
centroids = kmeans.cluster_centers_
Centers = {}
i = 0
for key in list(list_drzwi.keys()):
    Centers[key] = labels[i]
    i += 1
'''
directory2 = 'C:/AGA_studia/inzynierka/DATA/ivectory_sklejone_zdania_po_5_scentralizowane_grupami'
for key in list(IVectors.keys()):
    path = directory2 + "/" + key +".txt"
    ivector = IVectors[key]
    np.savetxt(path, ivector, newline=' ', fmt='%f')
'''

'''
fig = plt.figure()
ax = fig.add_subplot(111)
plt.boxplot([drzwi_mean, muz_mean, swiatlo_mean, temp_mean])
plt.title("Wykresy pudelkowe dla srednich i-vectorow centr dla komend (po 10)")
ax.set_xticklabels(["Otworz drzwi", "Wlacz muzyke", "Zapal swiatlo", "Zwieksz temeprature"])
plt.show()
'''
'''
plt.plot(drzwi_mean, "x")
plt.plot(muz_mean, "+")
plt.plot(swiatlo_mean, "*")
plt.plot(temp_mean, "o", markersize=3)
plt.title("Srednie i-vectory (20x) dla komend")
plt.legend(["Otworz drzwi", "Wlacz muzyke", "Zapal swiatlo", "Zwieksz temeprature"])
plt.show()
'''
print()