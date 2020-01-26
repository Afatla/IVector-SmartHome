from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder
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

def get_codes(Dict, codes=[], Codes=[]):
    osoby = get_people(Dict)

    for key in list(Dict.keys()):
        for i in range(len(osoby)):
            if len(key.split("_")) == 3:
                if key.split("_")[-3] == osoby[i]:
                    codes.append(i)
                    Codes.append(osoby[i])
            if len(key.split("_")) == 4:
                if key.split("_")[-4] + "_" + key.split("_")[-3] == osoby[i]:
                    codes.append(i)
                    Codes.append(osoby[i])
    return codes, Codes

def get_columns(n=600):
    lista = []
    for i in range(0, n):
        lista.append("el_"+str(i))
    return lista

def get_LDA(Dict, lista, osoby, n_test=2, n_components=3):
    lda = LDA(n_components=n_components)

    Results = {}
    for i in range(0, len(Dict)-n_test, n_test):
        Enroll = copy.deepcopy(Dict)
        l = list(Dict.keys())
        l.sort(key=lambda x: int(x.rsplit('_', 1)[-1]))
        test_class = l[i:i+n_test]
        test_list = []
        for j in range(len(test_class)):
            test_list.append(Dict[test_class[j]])
        for jj in range(len(test_class)):
            Enroll.__delitem__(test_class[jj])
        test_list = np.array(test_list)

        X = np.array(list(Enroll.values()))
        X = pd.DataFrame(X, columns=lista)
        codes, Codes = get_codes(Enroll, codes=[])
        y = pd.Categorical.from_codes(codes, osoby)
        df = X.join(pd.Series(y, name='class'))
        le = LabelEncoder()
        y = le.fit_transform(df['class'])

        X_lda = lda.fit_transform(X, y)

        osoby.sort()

        p = lda.predict(test_list)
        for ii in range(n_test):
            Results[test_class[ii]] = osoby[p[ii]]

    return Results


def scatter(X_lda, osoby, colmap={}):

    for i in range(0, len(X_lda[:, 0])):
        plt.scatter(X_lda[i, 0], X_lda[i, 1], color='g', s=5)
    plt.show()


directory = 'C:/AGA_studia/inzynierka/DATA/ivectory_centr_grupami'
IVectors, Drzwi, Muzyka, Swiatlo, Temp = load_ivectors(directory=directory)
Dict = Drzwi
osoby = get_people(Dict)
codes = get_codes(Dict)
count = 0
R_d = get_LDA(Dict=Dict, lista=get_columns(), osoby=osoby)
for key in list(R_d.keys()):
    if R_d[key] + "_" == key.split(key.split("_")[-2])[0]:
        count += 1
error_d = np.float((np.float(len(R_d))-np.float(count))/np.float(len(R_d)))
Dict = Muzyka
osoby = get_people(Dict)
codes = get_codes(Dict)
count = 0
R_m = get_LDA(Dict=Dict, lista=get_columns(), osoby=osoby)
for key in list(R_m.keys()):
    if R_m[key] + "_" == key.split(key.split("_")[-2])[0]:
        count += 1
error_m = np.float((np.float(len(R_m))-np.float(count))/np.float(len(R_m)))
Dict = Swiatlo
osoby = get_people(Dict)
codes = get_codes(Dict)
count = 0
R_s = get_LDA(Dict=Dict, lista=get_columns(), osoby=osoby)
for key in list(R_s.keys()):
    if R_s[key] + "_" == key.split(key.split("_")[-2])[0]:
        count += 1
error_s = np.float((np.float(len(R_s))-np.float(count))/np.float(len(R_s)))
Dict = Temp
osoby = get_people(Dict)
codes = get_codes(Dict)
count = 0
R_t = get_LDA(Dict=Dict, lista=get_columns(), osoby=osoby)
for key in list(R_t.keys()):
    if R_t[key] + "_" == key.split(key.split("_")[-2])[0]:
        count += 1
error_t = np.float((np.float(len(R_t))-np.float(count))/np.float(len(R_t)))
print()









