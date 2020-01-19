from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

directory = 'C:/AGA_studia/inzynierka/DATA/ivectory_sklejone_zdania_po_5'
IVectors = dict()
for file in os.listdir(directory):
    f = open(directory+"/"+file)
    ivector = f.read().split()
    for el in ivector:
        index = ivector.index(el)
        el = float(el)
        ivector[index] = el
    IVectors[file.split(".")[0]] = ivector
X = list(IVectors.values())
# codes
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
#columns
lista = []
for i in range(0, 600):
    lista.append("el_"+str(i))

X = np.array(list(IVectors.values()))
X = pd.DataFrame(X, columns=lista)
codes_names = ["drzwi", "muzyka", "swiatlo", "temp"]
#codes_names = ["drzwi", "swiatlo", "temp"]
y = pd.Categorical.from_codes(codes, codes_names)

df = X.join(pd.Series(y, name='class'))
le = LabelEncoder()
y = le.fit_transform(df['class'])
classifier = KNeighborsClassifier(n_neighbors=4)
classifier.fit(X, y)
y_pred = classifier.predict(X)
Centers = {}
i = 0
for key in (list(IVectors.keys())):
    Centers[key] = y_pred[i]
    i += 1
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

print(drzwi)
print(muz)
print(swiatlo)
print(temp)

print()