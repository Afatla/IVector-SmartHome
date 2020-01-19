import requests
import json

# sending wav file!
# endpoint = 'mfcc'
# url = "http://" + "localhost:5000/" + endpoint
#
# files = {'file' : open('plik.wav' , 'rb')}
#
# g = requests.post(url, files=files)
# print(g.headers)
# print(g.content.decode())

# sending string to get wanted matrix




# getting statistics
url = "http://localhost:5000/stats"

files = {'f' : open('plik.wav' , 'rb')}

g = requests.post(url, files=files, data={'type' : 'f'})

print(g.content.decode())