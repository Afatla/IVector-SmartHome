from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans


def load_ivectors(IVectors={}, Drzwi={}, Muzyka={}, Swiatlo={}, Temp={}, directory=""):
    for file in os.listdir(directory):
        f = open(directory + "/" + file)
        ivector = f.read().split()
        for el in ivector:
            index = ivector.index(el)
            el = float(el)
            ivector[index] = el
        if file.split("_")[-2] == "drzwi":
            Drzwi[file.split(".")[0]] = ivector
        elif file.split("_")[-2] == "muzyka":
            Muzyka[file.split(".")[0]] = ivector
        elif file.split("_")[-2] == "swiatlo":
            Swiatlo[file.split(".")[0]] = ivector
        elif file.split("_")[-2] == "temp":
            Temp[file.split(".")[0]] = ivector
        IVectors[file.split(".")[0]] = ivector
    return IVectors, Drzwi, Muzyka, Swiatlo, Temp

def get_codes(IVectors):
    codes = []
    for i in range(len(list(IVectors.values()))):
        if list(IVectors.keys())[i].split("_")[-2] == "drzwi":
            codes.append(0)
        if list(IVectors.keys())[i].split("_")[-2] == "muzyka":
            codes.append(1)
        if list(IVectors.keys())[i].split("_")[-2] == "swiatlo":
            codes.append(2)
        if list(IVectors.keys())[i].split("_")[-2] == "temp":
            codes.append(3)
        codes_names = ["drzwi", "muzyka", "swiatlo", "temp"]
    return codes, codes_names

def get_columns(n = 600):
    lista = []
    for i in range(0, n):
        lista.append("el_" + str(i))
    return lista

def get_LDA(IVectors):
    codes, codes_names = get_codes(IVectors)
    lista = get_columns()
    X = np.array(list(IVectors.values()))
    X = pd.DataFrame(X, columns=lista)
    y = pd.Categorical.from_codes(codes, codes_names)
    lda = LDA(n_components=3)
    df = X.join(pd.Series(y, name='class'))
    le = LabelEncoder()
    y = le.fit_transform(df['class'])
    X_lda = lda.fit_transform(X, y)
    i = 0
    X_lda_dict = {}
    for key in list(IVectors.keys()):
        X_lda_dict[key] = X_lda[i]
        i += 1
    
    return X_lda, X_lda_dict

def cluster(X_lda, IVectors):
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(X_lda)
    labels = kmeans.predict(X_lda)
    centroids = kmeans.cluster_centers_

    Centers = {}
    Drzwi = {}
    Swiatlo = {}
    Muzyka = {}
    Temp = {}
    i = 0
    for key in (list(IVectors.keys())):
        if key.split("_")[-2] == "drzwi":
            Drzwi[key] = labels[i]
        if key.split("_")[-2] == "muzyka":
            Muzyka[key] = labels[i]
        if key.split("_")[-2] == "swiatlo":
            Swiatlo[key] = labels[i]
        if key.split("_")[-2] == "temp":
            Temp[key] = labels[i]
        Centers[key] = labels[i]
        i += 1
    # do ktorych srodkow zostalo przypisane
    temp = [0, 0, 0, 0]
    drzwi = [0, 0, 0, 0]
    swiatlo = [0, 0, 0, 0]
    muz = [0, 0, 0, 0]
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

    return labels, Centers, centroids, Drzwi, Muzyka, Swiatlo, Temp, drzwi, muz, swiatlo, temp

def scatter(X_lda, centroids, IVectors, title, path_save):
    plt.xlabel('LD1')
    plt.ylabel('LD2')
    colmap_ = {"drzwi": 'r', "muzyka": 'g', "swiatlo": 'b', "temp": 'y'}

    for i in range(0, len(X_lda[:, 0])):
        plt.scatter(X_lda[i, 0], X_lda[i, 1],
                    color=colmap_[list(IVectors.keys())[i].split("_")[-2]], s=5)
    for i in range(len(centroids)):
        plt.scatter(*centroids[i], color='k', s=20)
    plt.title(title)
    plt.legend(["Wlacz muzyke", "Otworz drzwi", "Zwieksz temperature", "Zapal swiatlo"])
    if path_save:
        plt.savefig(path_save)
    plt.show()

path = 'C:/AGA_studia/inzynierka/DATA/ivectory_centr_grupami'
IVectors, Drzwi, Muzyka, Swiatlo, Temp = load_ivectors(directory=path)
X_lda, X_lda_dict = get_LDA(IVectors)

labels, Centers, centroids, Drzwi, Muzyka, Swiatlo, Temp, drzwi, muz, swiatlo, temp = cluster(X_lda, IVectors)
scatter(X_lda, centroids, IVectors, title="", path_save="C:\Users\Agnieszka\Desktop\inzynierka\lda\LDA_inz")
print()









