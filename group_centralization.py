import os
import numpy as np

Drzwi = {}
Muzyka = {}
Swiatlo = {}
Temp = {}
IVectors = dict()
directory = 'C:/AGA_studia/inzynierka/DATA/ivectory_10x_to_samo'
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

Drzwi_centr = {}
Muzyka_centr = {}
Swiatlo_centr = {}
Temp_centr = {}
IVectors_centr = {}
for key in list(Drzwi.keys()):
    Drzwi_centr[key] = np.array(Drzwi[key]) - mean_drzwi
    IVectors_centr[key] = Drzwi_centr[key]
for key in list(Muzyka.keys()):
    Muzyka_centr[key] = np.array(Muzyka[key]) - mean_muz
    IVectors_centr[key] = Muzyka_centr[key]
for key in list(Swiatlo.keys()):
    Swiatlo_centr[key] = np.array(Swiatlo[key]) - mean_swiatlo
    IVectors_centr[key] = Swiatlo_centr[key]
for key in list(Temp.keys()):
    Temp_centr[key] = np.array(Temp[key]) - mean_temp
    IVectors_centr[key] = Temp_centr[key]

directory = 'C:/AGA_studia/inzynierka/DATA/ivectory_10x_to_samo_centr_grupami/'
for key in list(IVectors_centr.keys()):
    np.savetxt(directory+key+".txt", IVectors_centr[key], newline=' ', fmt='%f')
print()
'''
directory = 'C:/AGA_studia/inzynierka/DATA/ivectory_sklejone_zdania_po_5'
for file in os.listdir(directory):
   f = open(directory + "/" + file)
   ivector = f.read().split()
   for el in ivector:
       index = ivector.index(el)
       el = float(el)
       ivector[index] = el
   if file.split("_")[-2] == "drzwi":
       Drzwi["po_5_"+file.split(".")[0]] = ivector
   elif file.split("_")[-2] == "muzyka":
       Muzyka["po_5_"+file.split(".")[0]] = ivector
   elif file.split("_")[-2] == "swiatlo":
       Swiatlo["po_5_"+file.split(".")[0]] = ivector
   elif file.split("_")[-2] == "temp":
       Temp["po_5_"+file.split(".")[0]] = ivector
   IVectors["po_5_"+file.split(".")[0]] = ivector
directory = 'C:/AGA_studia/inzynierka/DATA/ivectory_sklejone_zdania_po_10'
for file in os.listdir(directory):
   f = open(directory + "/" + file)
   ivector = f.read().split()
   for el in ivector:
       index = ivector.index(el)
       el = float(el)
       ivector[index] = el
   if file.split("_")[-2] == "drzwi":
       Drzwi["po_10_"+file.split(".")[0]] = ivector
   elif file.split("_")[-2] == "muzyka":
       Muzyka["po_10_"+file.split(".")[0]] = ivector
   elif file.split("_")[-2] == "swiatlo":
       Swiatlo["po_10_"+file.split(".")[0]] = ivector
   elif file.split("_")[-2] == "temp":
       Temp["po_10_"+file.split(".")[0]] = ivector
   IVectors["po_10_"+file.split(".")[0]] = ivector

mean_muz = np.mean(list(Muzyka.values()), axis=0)
mean_drzwi = np.mean(list(Drzwi.values()), axis=0)
mean_swiatlo = np.mean(list(Swiatlo.values()), axis=0)
mean_temp = np.mean(list(Temp.values()), axis=0)

Drzwi_centr = {}
Muzyka_centr = {}
Swiatlo_centr = {}
Temp_centr = {}
IVectors_centr = {}
for key in list(Drzwi.keys()):
   Drzwi_centr[key] = np.array(Drzwi[key]) - mean_drzwi
   IVectors_centr[key] = Drzwi_centr[key]
for key in list(Muzyka.keys()):
   Muzyka_centr[key] = np.array(Muzyka[key]) - mean_muz
   IVectors_centr[key] = Muzyka_centr[key]
for key in list(Swiatlo.keys()):
   Swiatlo_centr[key] = np.array(Swiatlo[key]) - mean_swiatlo
   IVectors_centr[key] = Swiatlo_centr[key]
for key in list(Temp.keys()):
   Temp_centr[key] = np.array(Temp[key]) - mean_temp
   IVectors_centr[key] = Temp_centr[key]

directory = 'C:/AGA_studia/inzynierka/DATA/ivectory_po_5_po_10_centr_grupami/'
for key in list(IVectors_centr.keys()):
   np.savetxt(directory+key+".txt", IVectors_centr[key], newline=' ', fmt='%f')

'''
