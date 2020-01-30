from scipy.spatial.distance import cosine as CDS
import os
import matplotlib
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
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

def get_people(IVectors, people=[]):
    for key in list(IVectors.keys()):
        if len(key.split("_")) == 3:
            if key.split("_")[-3] not in people:
                people.append(key.split("_")[-3])
        if len(key.split("_")) == 4:
            if key.split("_")[-4] + "_" + key.split("_")[-3] not in people:
                people.append(key.split("_")[-4] + "_" + key.split("_")[-3])
    return people

def filter_people(ivecs, Dict, Keys=[], X=[], p=None):
    if p == None:
        X = ivecs
        return X, Keys
    else:
        for i in range(len(ivecs)):
            if len(list(Dict.keys())[i].split("_")) == 3:
                if list(Dict.keys())[i].split("_")[-3] == p:
                    Keys.append(list(Dict.keys())[i])
                    X.append(ivecs[i])
            if len(list(Dict.keys())[i].split("_")) == 4:
                if list(Dict.keys())[i].split("_")[-4] + "_" + list(Dict.keys())[i].split("_")[-3] == p:
                    Keys.append(list(Dict.keys())[i])
                    X.append(ivecs[i])
        return X, Keys

def reduction_pca(Dict):
    X = list(Dict.values())
    pca = PCA(n_components=100)
    pca.fit(X)
    X_embedded = pca.fit_transform(X)
    i = 0
    X_embedded = list(X_embedded)
    for key in list(Dict.keys()):
        Dict[key] = X_embedded[i]
        i += 1
    return Dict

def get_distances(X, Keys, Dict):
    X, Keys = filter_people(X, Dict, Keys)
    Distances = {}
    for i in range(len(X)):
        for j in range(i, len(X)):
            Distances[Keys[i] + " " + Keys[j]] = CDS(X[i], X[j])

    return Distances

def cds_all(Dict, same_person_D = {}, diff_person_D = {}):
    # kazdy z kazdym

    people = get_people(Dict)
    ivecs = np.array(Dict.values())
    D = get_distances(ivecs, list(Dict.keys()), Dict)

    for key in list(D.keys()):
        k = key.split()
        if k[0].split("drzwi")[0] == k[1].split("drzwi")[0]:
            same_person_D[key] = D[key]

    for key in list(D.keys()):
        k = key.split()
        if k[0].split("drzwi")[0] != k[1].split("drzwi")[0]:
            diff_person_D[key] = D[key]
    drzwi_max_diff = np.max(np.array(diff_person_D.values()))
    drzwi_max_same = np.max(np.array(same_person_D.values()))
    drzwi_min_diff = np.min(np.array(diff_person_D.values()))
    drzwi_min_same = np.min(np.array(same_person_D.values())[np.array(same_person_D.values()) > 0])
    return drzwi_max_diff, drzwi_max_same, drzwi_min_diff, drzwi_min_same

def mean_cds(Dict, Distances_impostor={}, Distances_target={}):
    people = get_people(Dict)
    # CDS dla ivectorow roznych osob
    for i in range(len(list(Dict.keys()))):
        test = list(Dict.values())[i]
        test_k = list(Dict.keys())[i]
        for p in people:
            enroll = []
            for key in list(Dict.keys()):
                if key.split(key.split("_")[-2])[0] == p + "_" and key.split(key.split("_")[-2])[0] != test_k:
                    enroll.append(Dict[key])
            enroll = np.mean(np.array(enroll))
            Distances_impostor[test_k + " " + p] = CDS(test, enroll)
    # dla kazdej probki porownanie do sredniej z jego innych nagran
    idx = 0
    for key_t in list(Dict.keys()):
        Distances_target[key_t] = 0
        for key_e in list(Dict.keys()):
            if key_e.split(key_e.split("_")[-2])[0] == key_t.split(key_t.split("_")[-2])[0] and key_e != key_t:
                Distances_target[key_t] += CDS(Dict[key_t], Dict[key_e])
                idx += 1
        Distances_target[key_t] = float(Distances_target[key_t] / idx)
        idx = 0
    return Distances_impostor, Distances_target

def get_error_lists(Distances_impostor, Distances_target):
    mini = np.min(np.array(list(Distances_target.values())))
    MINI = np.min(np.array(list(Distances_impostor.values())))
    maxi = np.max(np.array(list(Distances_target.values())))
    MAXI = np.max(np.array(list(Distances_impostor.values())))
    if MINI < mini:
        mini = MINI
    if MAXI > maxi:
        maxi = MAXI
    FAR_list = []
    FRR_list = []
    for i in np.arange(mini, maxi, 1e-4):
        count = 0
        for value in list(Distances_target.values()):
            if value > i:
                count += 1
        FRR = np.float64(np.float64(count) / np.float64(len(Distances_target)))
        FRR_list.append(FRR)
        count = 0
        for value in list(Distances_impostor.values()):
            if value < i:
                count += 1
        FAR = np.float64(np.float64(count) / np.float64(len(Distances_impostor)))
        FAR_list.append(FAR)
    return FRR_list, FAR_list, mini, maxi

def far_frr_plot(mini, maxi, FAR_list, FRR_list):
    plt.plot(np.arange(mini, maxi, 1e-4), FAR_list)
    plt.plot(np.arange(mini, maxi, 1e-4), FRR_list)
    plt.legend(["FAR", "FRR"])
    plt.ylabel("Wartosc bledu")
    plt.xlabel("Prog akceptacji")
    plt.show()

def det_plot(FAR_list, FRR_list):
    axis_min = min(FAR_list[0], FRR_list[-1])
    fig, ax = plt.subplots()
    plt.plot(FAR_list, FRR_list)
    plt.xlabel("FAR")
    plt.ylabel("FRR")
    plt.yscale('log')
    plt.xscale('log')
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.show()

def calculate_and_plot(Dict):
    Distances_impostor, Distances_target = mean_cds(Dict)
    FRR_list, FAR_list, mini, maxi = get_error_lists(Distances_impostor, Distances_target)
    #far_frr_plot(mini, maxi, FAR_list, FRR_list)
    det_plot(FAR_list, FRR_list)
    return Distances_impostor, Distances_target, FRR_list, FAR_list, mini, maxi

path = 'C:/AGA_studia/inzynierka/DATA/ivectory_centr_grupami'
IVectors, Drzwi, Muzyka, Swiatlo, Temp = load_ivectors(directory=path)
Dict = Drzwi
calculate_and_plot(Dict)

Dict = Muzyka
