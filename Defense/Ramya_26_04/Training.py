import numpy as np
import matplotlib.pyplot as plt 
from tkinter.filedialog import askopenfilename
from tensorflow.keras.models import Sequential
import cv2
from skimage.io import imshow
from skimage.feature import hog

import os
import argparse
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense, Dropout
#from utils import makedirs
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# === GETTING INPUT SIGNAL


Trainfea_new = []

for ijklm in range(0,227):
    temp = ijklm+1
    print(temp)
    img = mpimg.imread('Training/IMG ('+str(temp)+').jpg')


# PRE-PROCESSING

    h1=224
    w1=224

    dimension = (w1, h1) 
    resized_image = cv2.resize(img,(h1,w1))

    SP = np.shape(resized_image)
   
    GRAY = resized_image

    MN_val = np.mean(GRAY)
    ST_val = np.std(GRAY)
    VR_val = np.var(GRAY)
    Features = [MN_val,ST_val,VR_val]
    Trainfea_new.append(Features)
    
    
import pickle
with open('Trainfea_new.pickle', 'wb') as f:
    pickle.dump(Trainfea_new, f) 