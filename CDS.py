from scipy.spatial.distance import cosine as CDS
import os
import matplotlib
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


def load_ivectors(IVectors={}, Drzwi={}, Muzyka={}, Swiatlo={}, Temp={}, directory="", filter=None):
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
    if filter != None:
        Drzwi_filter = {}
        Muzyka_filter = {}
        Swiatlo_filter = {}
        Temp_filter = {}
        people = get_people(IVectors)
        for p in people:
            idx = 0
            for key in list(Drzwi.keys()):
                if p+"_" == key.split("drzwi")[0]:
                    idx += 1
                    if idx < filter:
                        Drzwi_filter[key] = Drzwi[key]
        for p in people:
            idx = 0
            for key in list(Muzyka.keys()):
                if p+"_" == key.split("muzyka")[0]:
                    idx += 1
                    if idx < filter:
                        Muzyka_filter[key] = Muzyka[key]
        for p in people:
            idx = 0
            for key in list(Swiatlo.keys()):
                if p+"_" == key.split("swiatlo")[0]:
                    idx += 1
                    if idx < filter:
                        Swiatlo_filter[key] = Swiatlo[key]
        for p in people:
            idx = 0
            for key in list(Temp.keys()):
                if p + "_" == key.split("temp")[0]:
                    idx += 1
                    if idx < filter:
                        Temp_filter[key] = Temp[key]

        return IVectors, Drzwi_filter, Muzyka_filter, Swiatlo_filter, Temp_filter

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


def get_people(IVectors, osoby=[]):
    for key in list(IVectors.keys()):
        if len(key.split("_")) == 3:
            if key.split("_")[-3] not in osoby:
                osoby.append(key.split("_")[-3])
        if len(key.split("_")) == 4:
            if key.split("_")[-4] + "_" + key.split("_")[-3] not in osoby:
                osoby.append(key.split("_")[-4] + "_" + key.split("_")[-3])
    return osoby


def get_codes(Dict, people, codes=[]):
    for key in list(Dict.keys()):
        for i in range(len(people)):
            if len(key.split("_")) == 3:
                if key.split("_")[-3] == people[i]:
                    codes.append(i)

            if len(key.split("_")) == 4:
                if key.split("_")[-4] + "_" + key.split("_")[-3] == people[i]:
                    codes.append(i)

    return codes


def get_columns(n=600):
    lista = []
    for i in range(0, n):
        lista.append("el_"+str(i))
    return lista


def get_LDA(Dict):
    lista = get_columns()
    X_lda_Dict = {}
    lda = LDA(n_components=3)
    le = LabelEncoder()
    X = np.array(list(Dict.values()))
    X = pd.DataFrame(X, columns=lista)
    people = get_people(Dict)
    codes = get_codes(Dict, people)
    y = pd.Categorical.from_codes(codes, people)
    df = X.join(pd.Series(y, name='class'))
    y = le.fit_transform(df['class'])
    X_lda = lda.fit_transform(X, y)

    for i in range(len(list(Dict.keys()))):
        X_lda_Dict[list(Dict.keys())[i]] = X_lda[i]
    return X_lda_Dict

def far_frr_plot(mini, maxi, FAR_list, FRR_list, eer, idx,
                 Distances_impostor, Distances_target,):
    impostor_list = {}
    target_list = {}
    plt.plot(np.arange(mini, maxi, 1e-4), FAR_list)
    plt.plot(np.arange(mini, maxi, 1e-4), FRR_list)

    plt.yticks([0, 0.2, eer, 0.4, 0.6, 0.8, 1])

    x_ticks = [0.85, 0.9, np.arange(mini, maxi, 1e-4)[idx], 1.05, 1.1, 1.15]
    x = [0.85, 0.875, 0.9, 0.925, 0.95, np.arange(mini, maxi, 1e-4)[idx], 1, 1.025, 1.05, 1.075, 1.1, 1.125, 1.15]

    for i in x:
        impostor_list[i] = 0
        target_list[i] = 0
    for impostor in list(Distances_impostor.values()):
        for i in range(len(x)):
            if i != 0:
                if impostor <= x[i] and impostor > x[i-1]:
                    impostor_list[x[i]] += 1
            else:
                if impostor <= x[i]:
                    impostor_list[x[i]] += 1
    for target in list(Distances_target.values()):
        for i in range(len(x)):
            if i != 0:
                if target <= x[i] and target > x[i-1]:
                    target_list[x[i]] += 1
            else:
                if target <= x[i]:
                    target_list[x[i]] += 1
    suma_i = np.sum(np.array(impostor_list.values()))
    for key in list(impostor_list.keys()):
        impostor_list[key] = np.float(impostor_list[key])/np.float(suma_i)
    suma_t = np.sum(np.array(target_list.values()))
    for key in list(target_list.keys()):
        target_list[key] = np.float(target_list[key]) / np.float(suma_t)
    plt.xticks(x_ticks)
    plt.ylabel("Wartosc bledu")
    plt.xlabel("Prog akceptacji (wartosc CDS)")
    plt.grid()
    plt.bar(np.array(target_list.keys()), height=np.array(target_list.values()), width=0.02, alpha=0.6)

    plt.bar(np.array(impostor_list.keys()), height=np.array(impostor_list.values()), width=0.02, alpha=0.6)
    plt.legend(["FAR", "FRR", "target", "impostor"])

    plt.show()



def det_plot(FAR_list, FRR_list):
    for i in range(len(FRR_list)):
        if abs(FAR_list[i] - FRR_list[i]) < 0.002:
            idx = i
            eer = FRR_list[i]
    axis_min = min(FAR_list[0], FRR_list[-1])
    fig, ax = plt.subplots()

    plt.xlabel("FAR")
    plt.ylabel("FRR")
    plt.yscale('log')
    plt.xscale('log')
    plt.scatter(eer, eer, c="k", s=40)
    plt.plot(FAR_list, FRR_list)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.text(eer, eer+0.01, "EER="+str(eer))
    plt.grid()
    plt.show()

    return eer, idx


path = 'C:/AGA_studia/inzynierka/DATA/ivectory_centr_grupami'
IVectors, Drzwi, Muzyka, Swiatlo, Temp = load_ivectors(directory=path, filter=11)
Dict = Temp
Distances_impostor, Distances_target = mean_cds(Dict)
FRR_list, FAR_list, mini, maxi = get_error_lists(Distances_impostor, Distances_target)
for i in range(len(FRR_list)):
    if abs(FAR_list[i] - FRR_list[i]) < 0.0024:
        idx = i
        eer = FRR_list[i]
        print eer
    far_frr_plot(mini, maxi, FAR_list, FRR_list, eer, idx, Distances_impostor, Distances_target)

'''
fig, ax = plt.subplots()
for f in [4, 6, 9, 11]:
    IVectors, Drzwi, Muzyka, Swiatlo, Temp = load_ivectors(directory=path, filter=f)
    Dict = Temp
    Distances_impostor, Distances_target = mean_cds(Dict)
    FRR_list, FAR_list, mini, maxi = get_error_lists(Distances_impostor, Distances_target)
    if f == 4:
        eer = 0.333
    elif f == 6:
        eer = 0.29
    elif f == 9:
        eer = 0.296
    else:
        eer = 0.284
    plt.scatter(eer, eer, c="k", s=20)
    plt.plot(FAR_list, FRR_list)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel("FAR")
    plt.ylabel("FRR")
plt.legend(["3 i-vectory kezdej osoby", "5 i-vectorow kazdej osoby", "8 i-vectorow kazdej osoby", "10 i-vectorow kazdej osoby"])
plt.grid()
plt.show()
'''
print()
