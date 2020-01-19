import os
import matplotlib.pyplot as plt

def draw_boxplot(directory, fig_directory):
    drzwi = []
    muz = []
    swiatlo = []
    temp = []

    for file in os.listdir(directory):
        f = open(directory + "/" + file)
        ivector = f.read().split()
        for el in ivector:
            index = ivector.index(el)
            el = float(el)
            ivector[index] = el
        if file.split("_")[-2] == "drzwi":
            drzwi.append(ivector)
        elif file.split("_")[-2] == "muzyka":
            muz.append(ivector)
        elif file.split("_")[-2] == "swiatlo":
            swiatlo.append(ivector)
        elif file.split("_")[-2] == "temp":
            temp.append(ivector)
    xticks = []
    for i in range(0, 599, 10):
        xticks.append(i)
    lista = []
    boxes = []
    for j in range(len(drzwi[0])):
        for ivec in drzwi:
            lista.append(ivec[j])
        boxes.append(lista)
        lista = []

    fig = plt.figure(figsize=[100, 10])
    plt.boxplot(boxes)
    plt.xticks(xticks, xticks)
    plt.title(directory.split("/")[-1]+" drzwi")
    plt.rcParams.update({'font.size': 25})
    fig.savefig(fname=fig_directory+"/"+directory.split("/")[-1]+"drzwi.png")

    lista = []
    boxes = []
    for j in range(len(muz[0])):
        for ivec in muz:
            lista.append(ivec[j])
        boxes.append(lista)
        lista = []

    fig = plt.figure(figsize=[100, 10])
    plt.boxplot(boxes)
    plt.xticks(xticks, xticks)
    plt.title(directory.split("/")[-1]+" muzyka")
    plt.rcParams.update({'font.size': 25})
    fig.savefig(fname=fig_directory+"/"+directory.split("/")[-1]+"_muzyka.png")

    lista = []
    boxes = []
    for j in range(len(swiatlo[0])):
        for ivec in swiatlo:
            lista.append(ivec[j])
        boxes.append(lista)
        lista = []

    fig = plt.figure(figsize=[100, 10])
    plt.boxplot(boxes)
    plt.xticks(xticks, xticks)
    plt.title(directory.split("/")[-1]+" swiatlo")
    plt.rcParams.update({'font.size': 25})

    fig.savefig(fname=fig_directory+"/"+directory.split("/")[-1]+"_swiatlo.png")

    lista = []
    boxes = []
    for j in range(len(temp[0])):
        for ivec in temp:
            lista.append(ivec[j])
        boxes.append(lista)
        lista = []

    fig = plt.figure(figsize=[100, 10])
    plt.boxplot(boxes)
    plt.xticks(xticks, xticks)
    plt.title(directory.split("/")[-1]+" temp")
    plt.rcParams.update({'font.size': 25})

    fig.savefig(fname=fig_directory+"/"+directory.split("/")[-1]+"_temp.png")

fig_directory = 'C:/Users/Agnieszka/Desktop/inzynierka/boxploty'
directory = 'C:/AGA_studia/inzynierka/DATA/ivectory'
draw_boxplot(directory, fig_directory)
directory = 'C:/AGA_studia/inzynierka/DATA/ivectory_20x_to_samo'
draw_boxplot(directory, fig_directory)
directory = 'C:/AGA_studia/inzynierka/DATA/ivectory_centr_20x_to_samo'
draw_boxplot(directory, fig_directory)
directory = 'C:/AGA_studia/inzynierka/DATA/ivectory_centr'
draw_boxplot(directory, fig_directory)