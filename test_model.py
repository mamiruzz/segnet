
# coding: utf-8

# In[10]:


from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import itertools
#from helper import *
import os
import keras.models as models
from keras.layers import *
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from os.path import exists
import cv2
import numpy as np
import json
from scipy.misc import imresize
import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.image as mpimg

# In[11]:


with open('models/segnet.json') as model_file:
        model = models.model_from_json(model_file.read())


# In[12]:


model.compile(loss="categorical_crossentropy", optimizer='nadam', metrics=["accuracy"])
bst_weights_path="models/weights.hdf5"


# In[13]:


model.load_weights(bst_weights_path)


# In[14]:

#image_name = sys.argv[1]


#print(image_name)

# In[ ]:





# In[15]:


def my_norm(rgb):
    norm_img=np.zeros((rgb.shape[0], rgb.shape[1], 3),np.float32)
    b=rgb[:,:,0]
    g=rgb[:,:,1]
    r=rgb[:,:,2]

    norm_img[:,:,0]=cv2.equalizeHist(b)
    norm_img[:,:,1]=cv2.equalizeHist(g)
    norm_img[:,:,2]=cv2.equalizeHist(r)
    return (norm_img)


# In[16]:


############# COLOR CODE THE OUTPUTS ############
Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road_marking = [255,69,0]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]

label_colours = np.array([Sky, Building, Pole, Road, Pavement, Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])
labels = np.array(['Sky', 'Building', 'Pole', 'Road', 'Pavement', 'Tree', 'SignSymbol', 'Fence', 'Car', 'Pedestrian', 'Bicyclist', 'Unlabelled'])


def plot_images(predicted_mask):
    r = predicted_mask.copy()
    g = predicted_mask.copy()
    b = predicted_mask.copy()
    for l in range(0,11):
        r[predicted_mask==l]=label_colours[l,0]
        g[predicted_mask==l]=label_colours[l,1]
        b[predicted_mask==l]=label_colours[l,2]

    rgb = np.zeros((predicted_mask.shape[0], predicted_mask.shape[1], 3))
    rgb[:,:,0] = (r/255.0)#[:,:,0]
    rgb[:,:,1] = (g/255.0)#[:,:,1]
    rgb[:,:,2] = (b/255.0)#[:,:,2]
    return rgb






#train_data = np.load('./data/train_data.npy')


# In[17]:

image_folder = 'g_media_left'

segnet_folder = image_folder

csv_file = image_folder+'.csv'

#image_name = '8.jpg'

def map_label(pos):
    return str(labels[pos])


def GetFrameName(file_name, video_id, image_type):
    video_name = video_id+'-'
    image_extension = '.'+image_type
    frame = file_name.replace(video_name, '')
    frame = frame.replace(image_extension, '')
    return frame

def WriteResult(readdirectory, savedirectory):
    import os
    files = os.listdir(readdirectory)
    print(files)
    if os.path.isfile(csv_file)!=True:
        with open(csv_file, 'a') as f:
            f.write('fileName, frame, Sky, Building, Pole, Road, Pavement, Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled\n')
    j = 0
    while j<len(files):
        image = my_norm(cv2.imread(readdirectory+'/'+ files[j]))
        print(files[j])
        #image = my_norm(cv2.imread('custom_images/' + 2.jpg))
        #image = cv2.resize(image, (480,360))
        
        image = imresize(image, (360,480))
        image = np.rollaxis(image,2)
        #image = (np.rollaxis(my_norm(cv2.imread('custom_images/' + image_name)),2))
        
        
        
             
        
        image = np.expand_dims(image, axis=0)
        
        output = model.predict_proba(image)
        #print('output')
        #print(output[0])
        pred = plot_images(np.argmax(output[0],axis=1).reshape((360,480)))
        
        
        
        
        
        predicted_classes = np.argmax(output[0],axis=1).reshape((360,480))
        predicted_classes = np.reshape(predicted_classes, (np.product(predicted_classes.shape),))
        #print('argmax')
        total_numbers = len(predicted_classes)
       
        category_found = len(np.unique(predicted_classes))
        print("Total categories found: "+str(category_found))
        
        from collections import Counter
        d = Counter(predicted_classes)
        
        n, m = list(d.keys()), list(d.values())
        #print(len(labels))
        a = np.zeros(len(labels))
        #print(a)
        
        for val in n:
            #print(val)
            a[val] = 1
            
        #print(a)  
        
        i=0
        while(i<len(n)):
           a[n[i]] = round(m[i]/total_numbers, 4)
           i+=1
           
       
        with open(csv_file, 'a') as f:
            #f.write('fileName, frame, Sky, Building, Pole, Road, Pavement, Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled\n')
            f.write('{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n'.format(files[j], GetFrameName(files[j], readdirectory, 'jpg'), a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[8], a[9], a[10], a[11]))
        
        
        save_directory = savedirectory+'_segnet'
        
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        
        plt.imshow(pred)
        plt.savefig(save_directory+'/'+files[j])
            
#        plt.subplot(1,2,1)
#        plt.imshow(pred)
#        plt.subplot(1,2,2)
#        img=mpimg.imread(directory+'/'+ files[j])
#        plt.imshow(img)
        
        #plt.imshow(pred)
        plt.show()
        j+=1
    print('processing complete!')

WriteResult(image_folder, segnet_folder)



