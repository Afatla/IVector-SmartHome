from scipy.spatial.distance import cosine as CDS
import os
from sklearn.decomposition import PCA
import numpy as np

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

def get_distances(X, Keys, Dict):
    X, Keys = filter_people(X, Dict, Keys)
    Distances = {}
    for i in range(len(X)):
        for j in range(i, len(X)):
            Distances[Keys[i] + " " + Keys[j]] = CDS(X[i], X[j])

    return Distances

path = 'C:/AGA_studia/inzynierka/DATA/ivectory_10x_to_samo'
IVectors, Drzwi, Muzyka, Swiatlo, Temp = load_ivectors(directory=path)
Dict = Drzwi

# dla kazdej probki porownanie do sredniej z jego innych nagran
Distances_target = {}
idx = 0
for key_t in list(Dict.keys()):
    Distances_target[key_t] = 0
    for key_e in list(Dict.keys()):
        if key_e.split(key_e.split("_")[-2])[0] == key_t.split(key_t.split("_")[-2])[0] and key_e != key_t:
            Distances_target[key_t] += CDS(Dict[key_t], Dict[key_e])
            idx += 1
    Distances_target[key_t] = float(Distances_target[key_t]/idx)
    idx = 0
# dla kazdej probki porownanie do sredniej z pozostalych nagran
Distances_impostor = {}
idx = 0
for key_t in list(Dict.keys()):
    Distances_impostor[key_t] = 0
    for key_e in list(Dict.keys()):
        if key_e.split(key_e.split("_")[-2])[0] != key_t.split(key_t.split("_")[-2])[0]:
            Distances_impostor[key_t] += CDS(Dict[key_t], Dict[key_e])
            idx += 1
    Distances_impostor[key_t] = float(Distances_impostor[key_t]/idx)
    idx = 0

# LR
LR = {}
for key in list(Distances_target.keys()):
    LR[key] = Distances_target[key]/Distances_impostor[key]
count = 0
for value in list(LR.values()):
    if value > 1:
        count += 1

len(LR)
print()

'''
# CDS po redukcji wymiarow PCA kazdy z kazdym
X = list(Drzwi.values())
pca = PCA(n_components=50)
pca.fit(X)
X_embedded = pca.fit_transform(X)
D = get_distances(X=X_embedded, Keys=list(Drzwi.keys()), Dict=Drzwi)
same_person_D = {}
for key in list(D.keys()):
    k = key.split()
    if k[0].split("drzwi")[0] == k[1].split("drzwi")[0]:
        same_person_D[key] = D[key]

diff_person_D = {}
for key in list(D.keys()):
    k = key.split()
    if k[0].split("drzwi")[0] != k[1].split("drzwi")[0]:
        diff_person_D[key] = D[key]

drzwi_max_diff = np.max(np.array(diff_person_D.values()))
drzwi_max_same = np.max(np.array(same_person_D.values()))
drzwi_min_diff = np.min(np.array(diff_person_D.values()))
drzwi_min_same = np.min(np.array(same_person_D.values())[np.array(same_person_D.values()) > 0])
print()
'''
'''
# CDS dla ivectorow tych samych osob
Dict = Drzwi
people = get_people(Dict)
Distances_target = {}
for p in people:
    temp = {}
    for key in list(Dict.keys()):
        if key.split("drzwi")[0] == p + "_":
            temp[key] = Dict[key]
    test = list(temp.values())[0]
    enroll = np.mean(np.array(temp.values())[1:len(temp.values())])
    Distances_target[p] = np.degrees(np.arccos(1-CDS(test, enroll)))
# CDS dla ivectorow roznych osob
Distances_impostor = {}
for i in range(len(list(Dict.keys()))):
    test = list(Dict.values())[i]
    test_k = list(Dict.keys())[i]
    for p in people:
        enroll = []
        for key in list(Dict.keys()):
            if key.split("drzwi")[0] == p + "_" and key.split("drzwi")[0] != test_k:
                enroll.append(Dict[key])
        enroll = np.mean(np.array(enroll))
        Distances_impostor[test_k + " " + p] = np.degrees(np.arccos(1-CDS(test, enroll)))
print(np.min(np.array(Distances_impostor.values())))
'''
'''
# kazdy z kazdym
Dict = Drzwi
people = get_people(Dict)
ivecs = np.array(Dict.values())
D = get_distances(ivecs, list(Dict.keys()), Dict)

same_person_D = {}
for key in list(D.keys()):
    k = key.split()
    if k[0].split("drzwi")[0] == k[1].split("drzwi")[0]:
        same_person_D[key] = D[key]

diff_person_D = {}
for key in list(D.keys()):
    k = key.split()
    if k[0].split("drzwi")[0] != k[1].split("drzwi")[0]:
        diff_person_D[key] = D[key]

drzwi_max_diff = np.max(np.array(diff_person_D.values()))
drzwi_max_same = np.max(np.array(same_person_D.values()))
drzwi_min_diff = np.min(np.array(diff_person_D.values()))
drzwi_min_same = np.min(np.array(same_person_D.values())[np.array(same_person_D.values()) > 0])
Dict = Muzyka
people = get_people(Dict)
ivecs = np.array(Dict.values())
D = get_distances(ivecs, list(Dict.keys()), Dict)

same_person_D = {}
for key in list(D.keys()):
    k = key.split()
    if k[0].split("muzyka")[0] == k[1].split("muzyka")[0]:
        same_person_D[key] = D[key]

diff_person_D = {}
for key in list(D.keys()):
    k = key.split()
    if k[0].split("muzyka")[0] != k[1].split("muzyka")[0]:
        diff_person_D[key] = D[key]
muz_max_diff = np.max(np.array(diff_person_D.values()))
muz_max_same = np.max(np.array(same_person_D.values()))
muz_min_diff = np.min(np.array(diff_person_D.values()))
muz_min_same = np.min(np.array(same_person_D.values())[np.array(same_person_D.values()) > 0])
Dict = Swiatlo
people = get_people(Dict)
ivecs = np.array(Dict.values())
D = get_distances(ivecs, list(Dict.keys()), Dict)

same_person_D = {}
for key in list(D.keys()):
    k = key.split()
    if k[0].split("swiatlo")[0] == k[1].split("swiatlo")[0]:
        same_person_D[key] = D[key]

diff_person_D = {}
for key in list(D.keys()):
    k = key.split()
    if k[0].split("swiatlo")[0] != k[1].split("swiatlo")[0]:
        diff_person_D[key] = D[key]
sw_max_diff = np.max(np.array(diff_person_D.values()))
sw_max_same = np.max(np.array(same_person_D.values()))
sw_min_diff = np.min(np.array(diff_person_D.values()))
sw_min_same = np.min(np.array(same_person_D.values())[np.array(same_person_D.values()) > 0])
Dict = Temp
people = get_people(Dict)
ivecs = np.array(Dict.values())
D = get_distances(ivecs, list(Dict.keys()), Dict)

same_person_D = {}
for key in list(D.keys()):
    k = key.split()
    if k[0].split("temp")[0] == k[1].split("temp")[0]:
        same_person_D[key] = D[key]

diff_person_D = {}
for key in list(D.keys()):
    k = key.split()
    if k[0].split("temp")[0] != k[1].split("temp")[0]:
        diff_person_D[key] = D[key]
temp_max_diff = np.max(np.array(diff_person_D.values()))
temp_max_same = np.max(np.array(same_person_D.values()))
temp_min_diff = np.min(np.array(diff_person_D.values()))
temp_min_same = np.min(np.array(same_person_D.values())[np.array(same_person_D.values()) > 0])
print()
print (drzwi_max_diff + muz_max_diff + sw_max_diff + temp_max_diff)/4
print (drzwi_min_diff + muz_min_diff + sw_min_diff + temp_min_diff)/4
print (drzwi_min_same + muz_min_same + sw_min_same + temp_min_same)/4
print (drzwi_max_same + muz_max_same + sw_max_same + temp_max_same)/4
'''






