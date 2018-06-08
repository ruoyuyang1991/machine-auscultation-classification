import os
import matplotlib
matplotlib.use('Agg') # No pictures displayed
import pylab
import librosa
import librosa.display
import numpy as np
from PIL import Image
import os

inputPath = 'G:\sound_data\chatter_wav\chatter27wav'
outPath = 'F:\chatterdata\chatter27'

def processImage(filesource, destsource, name):
    #save_path = destsource + '1.jpeg'
    sig, fs = librosa.load(filesource + "/" + name)
    pylab.axis('off') # no axis
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
    S = librosa.feature.melspectrogram(y=sig, sr=fs)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    #S.save(destsource + "/" + name)
    pylab.savefig(destsource + "/" + name +'.jpeg', bbox_inches=None, pad_inches=0)
    pylab.close()

def run():
    os.chdir(inputPath)
    for i in os.listdir(os.getcwd()):
        processImage(inputPath, outPath, i)

run()