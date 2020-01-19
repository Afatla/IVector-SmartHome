import os
import numpy as np

directory = 'C:/AGA_studia/inzynierka/DATA/ivectory_sklejone_zdania_po_10'
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
mean_ivector = []
sum = 0
for i in range(len(X[0])):
    for ivector in X:
        sum = sum + ivector[i]
    mean_ivector.append(sum / len(X))
for ii in range(len(X)):
    ivec = X[ii]
    for i in range(len(mean_ivector)):
        ivec[i] = ivec[i] - mean_ivector[i]
    X[ii] = ivec

directory = 'C:/AGA_studia/inzynierka/DATA/ivectory_centr_sklejone_zdania_po_10'
idx = 0
for key in list(IVectors.keys()):
    path = directory+"/"+key+".txt"
    np.savetxt(path, X[idx], newline=' ', fmt='%f')
    idx += 1
