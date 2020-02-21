import os
from pydub import AudioSegment

directory = 'C:/AGA_studia/inzynierka/DATA/wavy_copy'
directory2 = 'C:/AGA_studia/inzynierka/DATA/wavy_1'
directory3 = 'C:/AGA_studia/inzynierka/DATA/wavy_2'
directory4 = 'C:/AGA_studia/inzynierka/DATA/wavy_3'
directory5 = 'C:/AGA_studia/inzynierka/DATA/wavy_4'
directory6 = 'C:/AGA_studia/inzynierka/DATA/wavy_5'

count = 0
for file in os.listdir(directory):
    count += 1
    sound = AudioSegment.from_wav(directory+"/"+file)
    if count == 1 or count == 2:
        combined_sound_1 = sound
        combined_sound_1.export(directory2 + "/" + file, format="wav")
    if count == 3 or count == 4:
        combined_sound_2 = sound
        combined_sound_2.export(directory3 + "/" + file, format="wav")
    if count == 5 or count == 6:
        combined_sound_3 = sound
        combined_sound_3.export(directory4 + "/" + file, format="wav")
    if count == 7 or count == 8:
        combined_sound_4 = sound
        combined_sound_4.export(directory5 + "/" + file, format="wav")
    if count == 9 or count == 10:
        combined_sound_5 = sound
        combined_sound_5.export(directory6 + "/" + file, format="wav")
    if count == 10:
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


