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


def get_LDA(Dict, lista, osoby, n_comp=3, n_test=2, Results = {}):
    l = list(Dict.keys())
    l.sort(key=lambda x: int(x.rsplit('_', 1)[-1]))
    for i in range(0, len(Dict)-n_test, n_test):
        lda = LDA(n_components=n_comp)
        Enroll = copy.deepcopy(Dict)
        test_class = l[i:i+n_test]
        test_list = []
        test_Dict = {}
        for ii in range(n_test):
            test_list.append(Dict[test_class[ii]])
            test_Dict[test_class[ii]] = test_list[ii]
            Enroll.__delitem__(test_class[ii])
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
        for n in range(n_test):
            Results[test_class[n]] = osoby[p[n]]
    return Results


def get_error(Dict, count=0):
    osoby = get_people(Dict)
    R = get_LDA(Dict=Dict, lista=get_columns(), osoby=osoby)
    for key in list(R.keys()):
        if R[key] + "_" == key.split(key.split("_")[-2])[0]:
            count += 1
    error = np.float((np.float(len(R))-np.float(count))/np.float(len(R)))
    return R, error


def scatter(X_lda, osoby, colmap={}):

    for i in range(0, len(X_lda[:, 0])):
        plt.scatter(X_lda[i, 0], X_lda[i, 1], color='g', s=5)
    plt.show()


directory = 'C:/AGA_studia/inzynierka/DATA/ivectory_20x_to_samo_centr_grupami'
IVectors, Drzwi, Muzyka, Swiatlo, Temp = load_ivectors(directory=directory)
Dict = Drzwi
R_d, error_d = get_error(Dict)

Dict = Muzyka










