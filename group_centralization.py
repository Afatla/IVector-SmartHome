import os
import numpy as np

Drzwi = {}
Muzyka = {}
Swiatlo = {}
Temp = {}
IVectors = dict()
directory = 'C:/AGA_studia/inzynierka/DATA/ivectory'
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
mean_muz = np.mean(list(Muzyka.values()), axis=0)
mean_drzwi = np.mean(list(Drzwi.values()), axis=0)
mean_swiatlo = np.mean(list(Swiatlo.values()), axis=0)
mean_temp = np.mean(list(Temp.values()), axis=0)
directory = 'C:/AGA_studia/inzynierka/DATA/ivectory_5'
D={}
M={}
S={}
T={}
for file in os.listdir(directory):
    f = open(directory + "/" + file)
    ivector = f.read().split()
    for el in ivector:
        index = ivector.index(el)
        el = float(el)
        ivector[index] = el
    if file.split("_")[-2] == "drzwi":
        D[file.split(".")[0]] = ivector
    elif file.split("_")[-2] == "muzyka":
        M[file.split(".")[0]] = ivector
    elif file.split("_")[-2] == "swiatlo":
        S[file.split(".")[0]] = ivector
    elif file.split("_")[-2] == "temp":
        T[file.split(".")[0]] = ivector
    IVectors[file.split(".")[0]] = ivector
Drzwi_centr = {}
Muzyka_centr = {}
Swiatlo_centr = {}
Temp_centr = {}
IVectors_centr = {}
for key in list(D.keys()):
    Drzwi_centr[key] = np.array(D[key]) - mean_drzwi
    IVectors_centr[key] = Drzwi_centr[key]
for key in list(M.keys()):
    Muzyka_centr[key] = np.array(M[key]) - mean_muz
    IVectors_centr[key] = Muzyka_centr[key]
for key in list(S.keys()):
    Swiatlo_centr[key] = np.array(S[key]) - mean_swiatlo
    IVectors_centr[key] = Swiatlo_centr[key]
for key in list(T.keys()):
    Temp_centr[key] = np.array(T[key]) - mean_temp
    IVectors_centr[key] = Temp_centr[key]

directory = 'C:/AGA_studia/inzynierka/DATA/ivectory_5_centr_grupami/'
for key in list(IVectors_centr.keys()):
    np.savetxt(directory+key+".txt", IVectors_centr[key], newline=' ', fmt='%f')
print()
