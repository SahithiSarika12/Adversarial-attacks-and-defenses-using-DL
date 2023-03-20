import tkinter as tk

import matplotlib.pyplot as plt 
from tkinter.filedialog import askopenfilename
from tensorflow.keras.models import Sequential
import cv2
from PIL import Image, ImageTk
from skimage.io import imshow,imread

from tkinter import *      
import pickle

from skimage import data

import tkinter

from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import os


from PIL import Image
from numpy import asarray

import numpy as np
from PIL import Image as im
import scipy.io
from skimage import color
from skimage import io
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from skimage import color
from skimage import io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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
from tkinter.filedialog import askopenfilename
import numpy as np
import matplotlib.pyplot as plt 
import cv2
from PIL import ImageTk, Image
from cv2 import *
import random
from skimage import color
from skimage import io
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
# Position text in frame

# Create a photoimage object of the image in the path

# Resize image to fit on button

# Position image on button

root = tk.Tk()

root.geometry("1080x600")

root.resizable(width=True, height=True)


img = None
resized_image = None

canvas = Canvas(root, width=1080, height=700)
import tkinter.messagebox


def openfn():
    global filename
    filename = askopenfilename(title='open')
    return filename

def write_slogan():
    
    global img
    x = openfn()
    img =Image.open(x)
    resized_ima = img.resize((300,300))

    img1 = ImageTk.PhotoImage(resized_ima)

    canvas.pack(pady=20)

# Add Images to Canvas widget

    canvas.create_image(210, 1, anchor=NW, image=img1)
    
    panel = Label(root, image=img1)
    panel.image = img1
    panel.pack()


def Preproce():
    global img
    global resized_image
    global imgGray
    import numpy as np
    import cv2
    from matplotlib import pyplot as plt   
    # PRE-PROCESSING
    h1=300
    w1=300

    resized_images = img.resize((h1,w1))
    resized_image = asarray(img.resize((h1, w1), Image.ANTIALIAS))
    resized_image1 = ImageTk.PhotoImage(resized_images)


    SP = np.shape(resized_image)
    try:
    
        Red = resized_image[:,:,0]
        Green = resized_image[:,:,1]
        Blue = resized_image[:,:,2]

    

        plt.imshow(Red)
        plt.title('RED IMAGE')
        plt.show()


        plt.imshow(Green)
        plt.title('GREEN IMAGE')
        plt.show()

        plt.imshow(Blue)
        plt.title('BLUE IMAGE')
        plt.show()

    except:
        None
    
        
    R, G, B = resized_image[:,:,0], resized_image[:,:,1], resized_image[:,:,2]
    imgGray = 0.2989 * R + 0.5870 * G + 0.1140 * B
    plt.imshow(imgGray, cmap='gray')
    plt.title('GRAY IMAGE')
    plt.show()


 
    img = mpimg.imread(filename)
    image_input = img

    image_input = color.rgb2gray(image_input)

    (x1, y) = image_input.shape
    image_input = image_input.astype(float) *255

    plt.figure()
    plt.imshow(image_input)
    plt.show()

    # print(image_input)

    mu, sigma = 0, 0.1  # mean and standard deviation
    key = np.random.normal(mu, sigma, (x1, y)) + np.finfo(float).eps
    # print(key)
    image_encrypted = image_input * key
    # imwrite('image_encrypted.jpg', image_encrypted * 255)

    plt.figure()
    plt.imshow(image_encrypted)
    plt.title('Attacked')
    plt.show()



    image_encrypted = image_encrypted.astype('uint8')

    images_arr = im.fromarray(image_encrypted)
    image_encryptedss = (images_arr.resize((300, 300), Image.ANTIALIAS))
    gray1 = ImageTk.PhotoImage(image_encryptedss)


    # canvas = Canvas(root, width=100, height=600)
    canvas.pack(pady=1)

# Add Images to Canvas widget

    img3 = canvas.create_image(512, 1, anchor=NW, image=gray1)
    panel1 = Label(root,image=gray1)
    
    panel1.pack(side=tk.TOP)

    panel1.image = gray1

    panel1.pack()
    
    plt.title('RESIZED IMAGE')
    plt.imshow(resized_image)
    plt.show()
    

    print('Preprocess completed...')

def gray():
    global resized_image
    global imgGray
    global filename
    global image_output
    GRAY = resized_image

    
    import numpy as np
    import cv2
    from matplotlib import pyplot as plt    
    img = mpimg.imread(filename)
    image_input = img

    image_input = color.rgb2gray(image_input)

    (x1, y) = image_input.shape
    image_input = image_input.astype(float) *255

    plt.figure()
    plt.imshow(image_input)
    plt.show()

    # print(image_input)

    mu, sigma = 0, 0.1  # mean and standard deviation
    key = np.random.normal(mu, sigma, (x1, y)) + np.finfo(float).eps
    # print(key)
    image_encrypted = image_input * key
    # imwrite('image_encrypted.jpg', image_encrypted * 255)

    plt.figure()
    plt.imshow(image_encrypted)
    plt.title('Attacked')
    plt.show()

    image_output = image_encrypted / key
    image_output /= 255.0
    # imwrite('image_output.jpg', image_output*255)

    plt.figure()
    plt.imshow(image_output)
    plt.show()

#    image_output = image_output.astype(np.uint8)
#
    h = 300
    w = 300
    
#    image_outputa = image_output.resize((h,w))
#    plt.figure()
#    plt.imshow(image_outputa)
#    plt.title('sdsdf')
#    plt.show()
#    
    
####################################################################
#
#    image_encryptedss = image_encrypted.resize((h,w))
#    image_encryptedaa = im.fromarray(image_encrypted)
    image_outputaas = im.fromarray(resized_image)
    print(1)
    image_encryptedss = (image_outputaas.resize((h, w), Image.ANTIALIAS))

    gray1s = ImageTk.PhotoImage(image_encryptedss)




    MN_val = np.mean(GRAY)
    ST_val = np.std(GRAY)
    VR_val = np.var(GRAY)

########################### CNN #######################################


    test_data = os.listdir('Samp/')
    train_data = os.listdir('Dataset/')

    dot= []
    labels = []
    for img in train_data:
        try:
            img_1 = plt.imread('Dataset/' + "/" + img)
            img_resize = cv2.resize(img_1,((50, 50)))
            dot.append(np.array(img_resize))
            labels.append(1)
        except:
            None
        
    for img in test_data:
        try:
            img_2 = plt.imread('Samp/'+ "/" + img)
            img_resize = cv2.resize(img_2,(50, 50))
            
            dot.append(np.array(img_resize))
            labels.append(0)
        except:
            None

    x_train, x_test, y_train, y_test = train_test_split(dot,labels,test_size = 0.2, random_state = 101)
    
    x_train1=np.zeros((len(x_train),50,50,3))
    for i in range(0,len(x_train)):
            x_train1[i,:,:]=x_train[i]
    
    x_test1=np.zeros((len(x_test),50,50,3))
    for i in range(0,len(x_test)):
            x_test1[i,:,:]=x_test[i]     
    
    
    
    
    from keras.utils import to_categorical
    
    model=Sequential()
    model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,1)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(500,activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(2,activation="softmax"))#2 represent output layer neurons 
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    y_train1=np.array(y_train)
    
    train_Y_one_hot = to_categorical(y_train1)
    test_Y_one_hot = to_categorical(y_test)
    
    
    Features = [MN_val,ST_val,VR_val]
    
    import pickle
    
    with open('Trainfea.pickle', 'rb') as fp:
         Train_features = pickle.load(fp)
         
         
    y_trains = np.arange(0,77)
    
        
    from sklearn.neighbors import KNeighborsClassifier
    
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(Train_features, y_trains)
    Class_KNN = neigh.predict([Features])
    
    import numpy as np
    import matplotlib.pyplot as plt
    Labeled = y_trains
    
    Labeled[0:11] = 1
    Labeled[11:21] = 2
    Labeled[21:33] = 3
    Labeled[33:44] = 4
    Labeled[44:56] = 5
    Labeled[56:67] = 6
    Labeled[67:77] = 7
    
    print('***********************************')
    
    Name = Labeled[Class_KNN]
    
    if Name == 1:
        
        print('Before Defence - Recognition Result')
        print('Identified - Sea Image')
        
        print(' ')
        print('Recognition Result - After Defence')
        print('Re-Labeled - Boat, Sea, Sky')
        
    elif Name == 2:
        
        print('Before Defence - Recognition Result')
        print('Identified - Car Image')
        
        print(' ')
        
        print('Recognition Result - After Defence')
        print('Re-Labeled - Traffic Image')
    
    elif Name == 3:
        
        print('Before Defence - Recognition Result')   
        print('Identified - Kitchen Image')
        
        print(' ')
        
        print('Recognition Result - After Defence')
        print('Re-Labeled - Mug an Jars')
    
    elif Name == 4:
        
        print('Before Defence - Recognition Result')
        print('Identified - Snow and Hill Image')
        print(' ')
        
        
        print('Recognition Result - After Defence')    
        print('Re-Labeled - Human (Couple Image)')
        
    elif Name == 5:
        
        print('Before Defence - Recognition Result')
        print('Identified - House Image')
        print(' ')
        
        print('Recognition Result - After Defence')    
        print('Re-Labeled - Visitors place')
        
    elif Name == 6:
        
        print('Before Defence - Recognition Result')
        print('Identified - Grass Land Image')
        print(' ')
        
        print('Recognition Result - After Defence')
        print('Re-Labeled - Human Photography')
        
    elif Name == 7:
        
        print('Before Defence - Recognition Result')
        print('Identified - Beach Image')
        print(' ')
        
        print('Recognition Result - After Defence')
        print('Re-Labeled - Human, Beach and Objects')
        
    print('***********************************')
     
    
    if Class_KNN == 1:
        print('Identified - Non Attack')
        
    else:
        print('Identified - Attack')
        print('Copy Move Attack is found...')
        Name = Labeled[Class_KNN]
        resized_image_ref = mpimg.imread('Samp/'+str(int(Name))+'.jpg')
        resized_image_ref = cv2.resize(resized_image_ref,(h,w))
    
        my_dpi = 60
    
        
        IBG = resized_image_ref - resized_image
    
    
        
    print('***********************************')
    
    
    
    from skimage import io, feature
    from scipy import ndimage
    import numpy as np
    
    def correlation_coefficient(patch1, patch2):
        product = np.mean((patch1 - patch1.mean()) * (patch2 - patch2.mean()))
        stds = patch1.std() * patch2.std()
        if stds == 0:
            return 0
        else:
            product /= stds
            return product
    
    #
    sh_row, sh_col = resized_image[:,:,0].shape
    #
    
    d = 1
    
    correlation = np.zeros_like(resized_image)
    
    correlation = correlation_coefficient(resized_image[:,:,0],
                                                        resized_image_ref[:,:,0])
    plt.show()
    
    print('Accuracy of Non Tamper = ',abs(correlation*100), ' %')
    
    
    correlation_t = correlation_coefficient(resized_image[:,:,0],
                                                        IBG[:,:,0])
    print('Accuracy Of Adversial / Tamper = ',abs(correlation_t*100), ' %')
    
    
    image_output = cv2.resize(image_output,(h,w))
    correlation_tz = correlation_coefficient(resized_image_ref[:,:,0],
                                                        image_output)
    print('Accuracy Of Defence Image = ',abs(correlation_tz*100), ' %')   
    
    
    ot.configure(text="Accuracy of Non Tamper: " + str(abs(correlation*100)))
    but.configure(text="Accuracy Of Adversial / Tamper: " + str(abs(correlation_t*100)))
    but1.configure(text="Accuracy Of Defence Image: " + str(abs(correlation_tz*100)))



    canvas.pack(pady=1)

# Add Images to Canvas widget

    img3 = canvas.create_image(210, 300, anchor=NW, image=gray1s)
    panel1 = Label(root,image=gray1s)
    
    panel1.pack(side=tk.TOP)

    panel1.image = gray1s

    panel1.pack()
    
    

    print('Segmentation completed...')
    

    
    
#def feat():
#    
#  
#    ot.configure(text="Features: " + str(Features))
#

#def classi():
    
 
    
#    but1.configure(text="Accuracy is: " +str(ACC*100)+ '  %')
       

 
    
def Close():
    root.destroy()
# root.pack()

btn = tk.Button(root, text='Input image',width=25, command=write_slogan)
# .pack()
btn.pack(side=tk.TOP)
btn.place(x=20, y=25)

btn = tk.Button(root, text='Attack Detection',width=25, command=Preproce)
btn.pack(side=tk.TOP)
btn.place(x=20, y=65) 

btn = tk.Button(root, text='Reconstruction',width=25, command=gray)
btn.pack(side=tk.TOP)
btn.place(x=20, y=105)

#btn = tk.Button(root, text='Reconstruction',width=25, command=feat)
#btn.pack(side=tk.TOP)
#btn.place(x=20, y=145)


#btn = tk.Button(root, text='Performance',width=25, command=classi)
#btn.pack(side=tk.TOP)
#btn.place(x=20, y=185)


btn = tk.Button(root, text='QUIT',width=25, command=Close)
btn.pack(side=tk.TOP)
btn.place(x=20, y=225)


ot = Label(root, text="",font=("Arial Bold", 10))
# ot.grid(column=1, row=19)
ot.place(x=512, y=300)


but = Label(root, text="",font=("Arial Bold", 10))
# ot.grid(column=1, row=19)
but.place(x=512, y=350)

but1 = Label(root, text="",font=("Arial Bold", 10))
# ot.grid(column=1, row=19)
but1.place(x=512, y=400)



root.mainloop()



