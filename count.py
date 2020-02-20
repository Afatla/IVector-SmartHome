import os
import numpy as np

directory = 'C:/AGA_studia/inzynierka/DATA/wavy_sklejone_zdania_po_10'
drzwi = 0
muzyka = 0
swiatlo = 0
temp = 0
for file in os.listdir(directory):
   if file.split("_")[-2] == "drzwi":
       drzwi += 1
   if file.split("_")[-2] == "muzyka":
       muzyka += 1
   if file.split("_")[-2] == "swiatlo":
       swiatlo += 1
   if file.split("_")[-2] == "temp":
       temp += 1

Count = {}
z_bliska = 0
z_daleka = 0
directory = 'C:/AGA_studia/inzynierka/DATA/wavy_mezczyzni'
for file in os.listdir(directory):
    if len(file.split("_")) == 3:
        Count[file.split("_")[0]+"_"+file.split("_")[1] + "_" + "z_daleka"] = 0
        Count[file.split("_")[0] + "_" + file.split("_")[1] + "_" + "z_bliska"] = 0
    if len(file.split("_")) == 4:
        Count[file.split("_")[0] + "_" + file.split("_")[1] + "_" + file.split("_")[2] + "_" + "z_daleka"] = 0
        Count[file.split("_")[0] + "_" + file.split("_")[1] + "_" + file.split("_")[2] + "_" + "z_bliska"] = 0
for file in os.listdir(directory):
    if int(file.split("_")[-1].split(".")[-2]) >= 6:
        if len(file.split("_")) == 3:
            Count[file.split("_")[0]+"_"+file.split("_")[1] + "_" + "z_daleka"] += 1
        if len(file.split("_")) == 4:
            Count[file.split("_")[0] + "_" + file.split("_")[1] + "_" + file.split("_")[2] + "_" + "z_daleka"] += 1
        z_daleka += 1
    if int(file.split("_")[-1].split(".")[-2]) < 6:
        if len(file.split("_")) == 3:
            Count[file.split("_")[0]+"_"+file.split("_")[1] + "_" + "z_bliska"] += 1
        if len(file.split("_")) == 4:
            Count[file.split("_")[0] + "_" + file.split("_")[1] + "_" + file.split("_")[2] + "_" + "z_bliska"] += 1
        z_bliska += 1

drzwi_z_bliska = 0
drzwi_z_daleka = 0
muzyka_z_bliska = 0
muzyka_z_daleka = 0
swiatlo_z_bliska = 0
swiatlo_z_daleka = 0
temp_z_bliska = 0
temp_z_daleka = 0
for key in list(Count.keys()):
    if key.endswith("drzwi_z_bliska"):
        drzwi_z_bliska += Count[key]
    if key.endswith("drzwi_z_daleka"):
        drzwi_z_daleka += Count[key]
    if key.endswith("muzyka_z_bliska"):
        muzyka_z_bliska += Count[key]
    if key.endswith("muzyka_z_daleka"):
        muzyka_z_daleka += Count[key]
    if key.endswith("swiatlo_z_bliska"):
        swiatlo_z_bliska += Count[key]
    if key.endswith("swiatlo_z_daleka"):
        swiatlo_z_daleka += Count[key]
    if key.endswith("temp_z_bliska"):
        temp_z_bliska += Count[key]
    if key.endswith("temp_z_daleka"):
        temp_z_daleka += Count[key]
print()



