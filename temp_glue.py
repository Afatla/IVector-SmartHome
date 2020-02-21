import os


d = 'C:/AGA_studia/inzynierka/DATA/ivectory_1_centr_grupami'
for file in os. listdir(d):
    os.rename(d+"/"+file, d+"/"+file.split(".txt")[0]+"5"+".txt")

print()
