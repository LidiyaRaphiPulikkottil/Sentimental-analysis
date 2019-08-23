from pydub import AudioSegment
import os

path1='/home/hp/Desktop/samples/'
path2='/home/hp/Desktop/samples/'

dirlist=os.listdir(path1)

for name in dirlist:
	path = path1+name
	sound = AudioSegment.from_wav(path)
	sound = sound.set_channels(1)
	sound.export(path2+name, format="wav")
