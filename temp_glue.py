import os
from pydub import AudioSegment
'''
d = 'C:/AGA_studia/inzynierka/DATA/ivectory_po_5_centr_grupami'
for file in os. listdir(d):
    os.rename(d+"/"+file, d+"/"+file[1:len(file)])
'''
directory = 'C:/AGA_studia/inzynierka/DATA/wavy_glue'
directory2 = 'C:/AGA_studia/inzynierka/DATA/wavy_sklejone_zdania_po_2'
directory3 = 'C:/AGA_studia/inzynierka/DATA/wavy_sklejone_zdania_po_8'

count = 0
for file in os.listdir(directory):
    count += 1
    sound = AudioSegment.from_wav(directory+"/"+file)
    if count == 1:
        combined_sound = sound
    if count > 1 and count <= 8:
        combined_sound += sound
    if count == 8:
        combined_sound.export(directory3 + "/" + file, format="wav")
        count = 0


directory = 'C:/AGA_studia/inzynierka/DATA/wavy'
directory2 = 'C:/AGA_studia/inzynierka/DATA/wavy_10x_to_samo'
for file in os.listdir(directory):
    combined_sound = AudioSegment.from_wav(directory+"/"+file)
    for i in range(9):
        sound = AudioSegment.from_wav(directory+"/"+file)
        combined_sound = combined_sound + sound
    combined_sound.export(directory2+"/"+file, format="wav")

directory = 'C:/AGA_studia/inzynierka/DATA/wavy'
directory2 = 'C:/AGA_studia/inzynierka/DATA/wavy_z_szumem_bialym'
szum = AudioSegment.from_wav("C:/AGA_studia/inzynierka/DATA/szum_bialy_500_ms.wav")
for file in os.listdir(directory):
    sound1 = AudioSegment.from_wav(directory + "/" + file)
    combined_sounds = szum + sound1 + szum
    combined_sounds.export(directory2+"/"+file, format="wav")


