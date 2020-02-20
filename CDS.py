from scipy.spatial.distance import cosine as CDS
import os
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import copy


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
        if file.split("_")[-2] == "muzyka":
            Muzyka[file.split(".")[0]] = ivector
        if file.split("_")[-2] == "swiatlo":
            Swiatlo[file.split(".")[0]] = ivector
        if file.split("_")[-2] == "temp":
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

def far_frr_plot(mini, maxi, FAR_list, FRR_list, eer, idx,
                 Distances_impostor, Distances_target,):
    impostor_list = {}
    target_list = {}
    plt.plot(np.arange(mini, maxi, 1e-4), FAR_list)
    plt.plot(np.arange(mini, maxi, 1e-4), FRR_list)

    plt.yticks([0, 0.2, eer, 0.4, 0.6, 0.8, 1])

    x_ticks = [0.2, 0.4, 0.6, np.arange(mini, maxi, 1e-4)[idx], 1.1, 1.2]
    x = [0.7, 0.8, 0.85, np.arange(mini, maxi, 1e-4)[idx], 1, maxi]
    plt.xlim(0.6, 1.3)
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
        if abs(FAR_list[i] - FRR_list[i]) <=0.007:
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
    plt.text(eer, eer+0.01, "EER=15%")
    plt.grid()
    plt.show()

    return eer, idx


def mean_cds_glue(Dict_test, Dict_train, Distances_impostor={}, Distances_target={}):
    people = get_people(Dict_train)
    D_test = copy.deepcopy(Dict_test)
    # CDS dla ivectorow roznych osob
    for i in range(len(list(Dict_test.keys()))):
        test = list(Dict_test.values())[i]
        test_k = list(Dict_test.keys())[i]
        for p in people:
            enroll = []
            for key in list(Dict_train.keys()):
                if key.split(key.split("_")[-2])[-2] == p + "_" and key.split(key.split("_")[-2])[0] != test_k.split(test_k.split("_")[-2])[0]:
                    enroll.append(Dict_train[key])
            enroll = np.mean(np.array(enroll))
            Distances_impostor[test_k + " " + p] = CDS(test, enroll)
    # dla kazdej probki porownanie do sredniej z jego innych nagran
    idx = 0
    for key_t in list(D_test.keys()):
        Distances_target[key_t] = 0
        for key_e in list(Dict_train.keys()):
            if key_e.split(key_e.split("_")[-2])[0] == key_t.split(key_t.split("_")[-2])[0] and key_e != key_t:
                Distances_target[key_t] += CDS(D_test[key_t], Dict_train[key_e])
                idx += 1
        if idx != 0:
            Distances_target[key_t] = float(Distances_target[key_t] / idx)
        idx = 0
    return Distances_impostor, Distances_target

path_1 = 'C:/AGA_studia/inzynierka/DATA/ivectory_sklejone_zdania_po_8_centr_grupami'
IVectors, Drzwi, Muzyka, Swiatlo, Temp = load_ivectors(directory=path_1, filter=None)
Dict_train = copy.deepcopy(Drzwi)
path_2 = 'C:/AGA_studia/inzynierka/DATA/ivectory_sklejone_zdania_po_2_centr_grupami'
IVectors_test, Drzwi_test, Muzyka_test, Swiatlo_test, Temp_test = load_ivectors(directory=path_2, filter=None)
Dict_test = copy.deepcopy(Drzwi_test)
Distances_impostor, Distances_target = mean_cds_glue(Dict_test, Dict_train)
FRR_list, FAR_list, mini, maxi = get_error_lists(Distances_impostor, Distances_target)
for i in range(len(FRR_list)):
    if abs(FAR_list[i] - FRR_list[i]) <= 0.001:
        idx = i
        eer = FRR_list[i]
        print eer
    far_frr_plot(mini, maxi, FAR_list, FRR_list, eer, idx, Distances_impostor, Distances_target)


print()
