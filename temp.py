import os
import matplotlib.pyplot as plt



directory = 'C:/AGA_studia/inzynierka/DATA/ivectory'
IVectors = dict()
for file in os.listdir(directory):
    f = open(directory+"/"+file)
    ivector = f.read().split()
    for el in ivector:
        index = ivector.index(el)
        el = float(el)
        ivector[index] = el
    print(max(ivector))
    print(ivector.index(max(ivector)))
    print(file)
    IVectors[file.split(".")[0]] = ivector


directory = 'C:/AGA_studia/inzynierka/DATA/ivectory_centr_20x_to_samo'
directory2 = 'C:/AGA_studia/inzynierka/DATA/ivectory_centr_20x_to_samo_drzwi_swiatlo'
for file in os.listdir(directory):
    if file.split("_")[-2] == "drzwi" or file.split("_")[-2] == "temp":
        f = open(directory+"/"+file)
        x = f.read()
        f = open(directory2+"/"+file, "w")
        f.write(x)


