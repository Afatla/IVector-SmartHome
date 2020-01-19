from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder

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

def get_codes(Dict, codes=[]):
    osoby = get_people(Dict)
    for key in list(Dict.keys()):
        for i in range(len(osoby)):
            if len(key.split("_")) == 3:
                if key.split("_")[-3] == osoby[i]:
                    codes.append(i)
            if len(key.split("_")) == 4:
                if key.split("_")[-4] + "_" + key.split("_")[-3] == osoby[i]:
                    codes.append(i)
    return codes

def get_columns():
    lista = []
    for i in range(0, 600):
        lista.append("el_"+str(i))
    return lista

def get_LDA(Dict, lista, osoby):
    lda = LDA(n_components=3)
    X = np.array(list(Dict.values()))
    X = pd.DataFrame(X, columns=lista)
    y = pd.Categorical.from_codes(get_codes(Dict, codes=[]), osoby)
    df = X.join(pd.Series(y, name='class'))
    le = LabelEncoder()
    y = le.fit_transform(df['class'])
    X_lda = lda.fit_transform(X, y)
    proba = lda.predict_proba(X)
    osoby.sort()

    proba_df = pd.DataFrame(proba, columns=osoby)
    proba_df = proba_df.join(pd.Series(Dict.keys(), name="class"))
    return X_lda, X, lda

def scatter(X_lda, osoby, colmap={}):

    for i in range(0, len(X_lda[:, 0])):
        plt.scatter(X_lda[i, 0], X_lda[i, 1], color='g', s=5)
    plt.show()


directory = 'C:/AGA_studia/inzynierka/DATA/ivectory_centr_grupami'
IVectors, Drzwi, Muzyka, Swiatlo, Temp = load_ivectors(directory=directory)

osoby = get_people(Drzwi)
codes = get_codes(Drzwi)
X_lda, X, lda = get_LDA(Dict=Drzwi, lista=get_columns(), osoby=osoby)

print()










