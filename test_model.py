
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

# In[11]:


with open('models/segnet.json') as model_file:
        model = models.model_from_json(model_file.read())


# In[12]:


model.compile(loss="categorical_crossentropy", optimizer='nadam', metrics=["accuracy"])
bst_weights_path="models/weights.hdf5"


# In[13]:


model.load_weights(bst_weights_path)


# In[14]:

image_name = sys.argv[1]
print(image_name)
#image_name = '2.jpg'


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


image = my_norm(cv2.imread('custom_images/' + image_name))
#image = my_norm(cv2.imread('custom_images/' + 2.jpg))
#image = cv2.resize(image, (480,360))

image = imresize(image, (360,480))
image = np.rollaxis(image,2)
#image = (np.rollaxis(my_norm(cv2.imread('custom_images/' + image_name)),2))



# In[18]:


image = np.expand_dims(image, axis=0)

output = model.predict_proba(image)
pred = plot_images(np.argmax(output[0],axis=1).reshape((360,480)))

objects_detected = len(np.unique(pred))
print('Total Object Types detected:', objects_detected)

dict = {'image_name': image_name, 'objects_detected': objects_detected}
#Change w to a if you want to append 
f = open('output.txt','a')
f.write(str(dict))
f.write('\n')
f.close()

#df = pd.DataFrame()
#df ['img_name'] = str(image_name)
#df['objects_detected'] = 'abc'
#df.to_csv('test.tsv', sep='\t', index=False)

plt.imshow(pred)
plt.show()
# In[ ]:




