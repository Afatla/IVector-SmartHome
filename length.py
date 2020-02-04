import soundfile as sf
import os
path = 'C:/AGA_studia/inzynierka/DATA/wavy_sklejone_zdania_po_5'
duration = 0
idx = 0
for fname in os.listdir(path):
    f = sf.SoundFile(path+"/"+fname)
    duration += len(f) / f.samplerate
    idx += 1
mean = duration/idx

path = 'C:/AGA_studia/inzynierka/DATA/wavy_sklejone_zdania_po_10'
duration = 0
idx = 0
for fname in os.listdir(path):
    f = sf.SoundFile(path+"/"+fname)
    duration += len(f) / f.samplerate
    idx += 1
mean = duration/idx
print
