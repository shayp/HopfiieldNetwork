from numpy import *
import matplotlib.pyplot as plt
from matplotlib.image import *
from PIL import Image
from scipy import *
import glob

def getImageArrayFromFolder(folder_name, img_type, xSize, ySize):
    image_list = []
    for filename in glob.glob(folder_name +'/*.' + img_type):
        im=Image.open(filename).convert('L')
        im= im.convert('1') # convert image to black and white
        #im = im.resize([xSize, ySize], Image.ANTIALIAS)
        im = im.resize([xSize, ySize], Image.ANTIALIAS)
        image_list.append(im)
    return image_list


def update_images_offset(images, xoffset, yoffset):
    for image in images:
        image = image.offset(xoffset,yoffset)
    return images

def fromImageArrayToPatternArray(images):
    vectorOfPatterns = []

    for image in images:
        vals=array(image.getdata())
        patternVector = img_to_pattern(vals)
        vectorOfPatterns += [patternVector]
    return np.asarray(vectorOfPatterns)


def img_to_pattern(object):
    return array([-1 if c < 128 else +1 for c in object])

def pattern_to_img(object):
    return array([255 if c > 0 else 0 for c in object])

def display(pattern, index, xSize,ySize):
    plt.figure(index)
    plt.imshow(pattern.reshape((xSize,ySize)),cmap=cm.binary)