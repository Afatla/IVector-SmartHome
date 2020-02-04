import os
import numpy as np
import matplotlib.pyplot as plt


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


directory = 'C:/AGA_studia/inzynierka/DATA/ivectory'
IVectors, Drzwi, Muzyka, Swiatlo, Temp = load_ivectors(directory=directory)
mean_muz = np.mean(list(Muzyka.values()), axis=0)
mean_drzwi = np.mean(list(Drzwi.values()), axis=0)
mean_swiatlo = np.mean(list(Swiatlo.values()), axis=0)
mean_temp = np.mean(list(Temp.values()), axis=0)
m = np.mean([mean_drzwi, mean_muz, mean_swiatlo, mean_temp], axis=0)
mean_muz = mean_muz - m
mean_drzwi = mean_drzwi - m
mean_swiatlo = mean_swiatlo - m
mean_temp = mean_temp - m
plt.plot(mean_drzwi, 'r+', markersize=3)
plt.plot(mean_muz, 'y*', markersize=3)
plt.plot(mean_swiatlo, 'bx', markersize=3)
plt.plot(mean_temp, 'go', markersize=2)
plt.grid()
plt.xticks([0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600])
plt.yticks([-0.06, -0.05, -0.04, -0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03, 0.04, 0.05])
plt.legend(["Otworz drzwi", "Wlacz muzyke", "Zapal swiatlo", "Zwieksz temperature"])
plt.show()
print