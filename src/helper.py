from hashlib import sha3_256
import cv2 
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.colors import LightSource
import math 
import scipy 
import skimage
import os
import glob 

def readimage_int(path):
    return cv2.imread(path).astype(np.int32)

def readimage(path):
    return cv2.imread(path).astype(np.float32) / 255.0

def writeimage(path, image):
    return cv2.imwrite(path, (np.clip(image,0,1) * 255).astype(np.uint8))

def crop(r, I):
    return I[r[0]:r[1], r[2]:r[3]]

def convert(image):
    return cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

def gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def showimage(image, cm='gray'):
    plt.imshow(convert(image), cmap=cm)
    plt.show()

def showimage_raw(image, cm='gray'):
    plt.imshow(image, cmap=cm)
    plt.show()

def showmaker(x, b = 50):
    plt.hist(x, bins = b)
    plt.gca().set(title='Frequency Histogram', ylabel='Frequency');
    plt.show()

inputpath = '../data/input/'
outputpath = '../data/output/'
