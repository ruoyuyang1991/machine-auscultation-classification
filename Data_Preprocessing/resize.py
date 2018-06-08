from PIL import Image
import os

inputPath = 'F:\\chatterdata\\chatter27'
outPath = 'F:\\chatterdata\\chatter27_150'


def processImage(filesource, destsource, name):

    im = Image.open(filesource + "/" + name)
    im = im.resize((150,150))
    im.save(destsource + "/" + name)


def run():

    os.chdir(inputPath)
    for i in os.listdir(os.getcwd()):
        processImage(inputPath, outPath, i)
run()
