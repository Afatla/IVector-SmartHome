import pandas as pd
import numpy as np
import os
import copy

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

def get_people(IVectors, osoby=[]):
    for key in list(IVectors.keys()):
        if len(key.split("_")) == 3:
            if key.split("_")[-3] not in osoby:
                osoby.append(key.split("_")[-3])
        if len(key.split("_")) == 4:
            if key.split("_")[-4] + "_" + key.split("_")[-3] not in osoby:
                osoby.append(key.split("_")[-4] + "_" + key.split("_")[-3])
    return osoby

def warp2us(ivecs, lda, lda_mu):
    """ i-vector pre-processing
        This function applies a global LDA, mean subtraction, and length
        normalization.
    """
    ivecs = ivecs.dot(lda) - lda_mu
    ivecs /= np.sqrt((ivecs ** 2).sum(axis=1)[:, np.newaxis])
    return ivecs

def bilinear_plda(Lambda, Gamma, c, k, Fe, Ft):
    """ Performs a full PLDA scoring
    """
    #out = np.empty((Fe.shape[0], Ft.shape[0]), dtype=Lambda.dtype)

    out = np.dot(Fe.dot(Lambda), Ft.T)
    out += (np.sum(Fe.dot(Gamma) * Fe, 1) + Fe.dot(c))[:, np.newaxis]
    out += (np.sum(Ft.dot(Gamma) * Ft, 1) + Ft.dot(c))[np.newaxis, :] + k
    return out


def get_PLDA(Dict, test):
    plda_model_dir = "../models/backend"
    lda_file = plda_model_dir + '/backend.LDA.txt.gz'
    mu_file = plda_model_dir + '/backend.mu_train.txt.gz'
    Gamma_file = plda_model_dir + '/backend.PLDA.Gamma.txt.gz'
    Lambda_file = plda_model_dir + '/backend.PLDA.Lambda.txt.gz'
    c_file = plda_model_dir + '/backend.PLDA.c.txt.gz'
    k_file = plda_model_dir + '/backend.PLDA.k.txt.gz'
    lda = np.loadtxt(lda_file, dtype=np.float32)
    mu = np.loadtxt(mu_file, dtype=np.float32)
    Gamma = np.loadtxt(Gamma_file, dtype=np.float32)
    Lambda = np.loadtxt(Lambda_file, dtype=np.float32)
    c = np.loadtxt(c_file, dtype=np.float32)
    k = np.loadtxt(k_file, dtype=np.float32)

    print 'Transforming and normalizing i-vectors'
    X = np.array(Dict.values())
    Y = np.array(test.values())
    enroll_ivec = X
    enroll_ivec = warp2us(enroll_ivec, lda, mu)
    test_ivec = Y
    rows = list(Dict.keys())
    columns = list(test.keys())
    test_ivec = warp2us(test_ivec, lda, mu)

    print 'Computing PLDA score'
    s = bilinear_plda(Lambda, Gamma, c, k, enroll_ivec, test_ivec)
    s = pd.DataFrame(s[:,-1], columns=[columns[0]])
    s = s.join(pd.Series(rows, name="class"))

    return s

directory = 'C:/AGA_studia/inzynierka/DATA/ivectory_centr_grupami'
IVectors, Drzwi, Muzyka, Swiatlo, Temp = load_ivectors(directory=directory)
Scores = {}
people = get_people(Muzyka)
'''
filtered = copy.deepcopy(Muzyka)
for i in range(len(people)):
    p = people[i]
    for key in list(filtered.keys()):
        if len(key.split("_")) == 3:
            if key.split("_")[-3] != p:
                filtered.pop(key)
        if len(key.split("_")) == 4:
            if key.split("_")[-4] + "_" + key.split("_")[-3] != p:
                filtered.pop(key)
    if len(filtered) > 1:
        s = PLDA(filtered)
        Scores[p] = s
'''
'''

p = people[0]
filtered = copy.deepcopy(Muzyka)
for key in list(filtered.keys()):
    if len(key.split("_")) == 3:
        if key.split("_")[-3] != p:
            filtered.pop(key)
    if len(key.split("_")) == 4:
        if key.split("_")[-4] + "_" + key.split("_")[-3] != p:
            filtered.pop(key)

temp = {}
Scores = {}
for key in list(Muzyka.keys()):
    if key.split("_")[-3] != p:
        temp[key] = Muzyka[key]
for i in range(len(temp)):
    test = {}
    for idx in range(len(filtered)):
        test[list(temp.keys())[i]+"_row_"+str(idx)] = temp[list(temp.keys())[i]]
    s = PLDA_impostor(filtered, test)
    Scores[list(temp.keys())[i]] = s
'''
Scores = {}
for i in range(len(people)):
    p = people[i]
    filtered = copy.deepcopy(Muzyka)
    for key in list(filtered.keys()):
        if len(key.split("_")) == 3:
            if key.split("_")[-3] != p:
                filtered.pop(key)
        if len(key.split("_")) == 4:
            if key.split("_")[-4] + "_" + key.split("_")[-3] != p:
                filtered.pop(key)
    test_row = 0
    test = {}
    idx = 0
    for j in range(len(filtered)-1):
        test[list(filtered.keys())[test_row] + "_" + str(idx)] = \
            filtered[list(filtered.keys())[test_row]]
        idx += 1
    filtered.pop(list(filtered.keys())[test_row])
    s = get_PLDA(filtered, test)
    Scores[p] = s


print()

