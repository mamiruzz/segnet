
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

image_name = 'custom_images/8.jpg'
print(image_name)

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


image = my_norm(cv2.imread(image_name))
#print(image)
#image = my_norm(cv2.imread('custom_images/' + 2.jpg))
#image = cv2.resize(image, (480,360))

image = imresize(image, (360,480))
image = np.rollaxis(image,2)
#image = (np.rollaxis(my_norm(cv2.imread('custom_images/' + image_name)),2))



# In[18]:


image = np.expand_dims(image, axis=0)

output = model.predict_proba(image)
#print('output')
#print(output[0])
pred = plot_images(np.argmax(output[0],axis=1).reshape((360,480)))
#print('pred')
#print(pred)
#array = np.argmax(output[0],axis=1).reshape((360,480))
#array = pred
#from PIL import Image
#print(array)
#
#img = Image.fromarray(array.astype('uint8'))
#img.show()
#plt.imshow(img)

#plt.plot(np.argmax(output[0],axis=1).reshape((360,480)))

#plt.colorbar(orientation='vertical')
#plt.show()


#import numpy_indexed as npi
#index = np.arange(pred.size) # array_3d[0].size
#(value, index), count = npi.count((pred.flatten(), index))
#a = np.round(pred)
#a = pred
data = pred.ravel()

import json
#print(json.dumps(data, default=lambda x: list(x), indent=4))
a = json.dumps(data, default=lambda x: list(x), indent=4)
#item_dict = json.loads(json.dumps(data, default=lambda x: list(x), indent=4))
def unique_count(a):
    unique, inverse = np.unique(a, return_inverse=True)
    count = np.zeros(len(unique), np.int)
    np.add.at(count, inverse, 1)
    return np.vstack(( unique, count)).T

#print(unique_count(data))
b = np.unique(pred)
c = 360*480*3
print(c)
print(b)
print(np.sum(np.where(data==b[0]))/c)
print(np.sum(np.where(data==b[1]))/c)
print(np.sum(np.where(data==b[2]))/c)
print(np.sum(np.where(data==b[3]))/c)
print(np.sum(np.where(data==b[4]))/c)
print(np.sum(np.where(data==b[5]))/c)
print(np.sum(np.where(data==b[6]))/c)


#print(len(item_dict['result'][0]['run']))
with open('test.txt', 'w') as f: f.write(json.dumps(data, default=lambda x: list(x), indent=4))
print(data.shape)

#unique, counts = np.unique(a, return_counts=True)


#print(np.asarray((unique, counts)).T)
#print(np.bincount(a.astype(np.int64)))

# =============================================================================
# print(len(pred))
# #b = np.reshape(pred, (1,np.product(pred.shape)))
# b =pred.ravel()
# b= np.asarray(b, dtype=int)
# c = np.bincount(b)
# print(len(b))
# print(b.shape)
# print(c)
# =============================================================================
# =============================================================================
# import collections
# #a = [1,1,1,1,2,2,2,2,3,3,4,5,5]
# #counter=collections.Counter(a)
# counter=collections.Counter(pred)
# print(counter)
# =============================================================================

# =============================================================================
# img = Image.fromarray(array)
# img.save('testrgb.png')
# img=mpimg.imread('testrgb.png')
# plt.imshow(img)
# plt.show()
# =============================================================================

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

plt.subplot(1,2,1)
plt.imshow(pred)
plt.subplot(1,2,2)
img=mpimg.imread(image_name)
plt.imshow(img)

#plt.imshow(pred)
plt.show()
# In[ ]:




