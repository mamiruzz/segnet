# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 20:20:53 2018

@author: mamiruzz
"""




WIDTH = 640
HEIGHT = 480

from PIL import Image
import os
import sys

#path = "C:\\1050215\\videos\\test\\"

path  = 'C:\\1050215\\videos\\1050215-R-2-old\\' #path to img source folder
output = 'C:\\1050215\\videos\\1050215-R-2\\' #path to img destination folder
dirs = os.listdir(path)

def resize(path, output):
    for item in dirs:
        if os.path.isfile(path + item):
            im = Image.open(path + item)
            f, e = os.path.splitext(path + item)
            print(f)
            file_name = os.path.basename(item)
            print(file_name)
            imResize = im.resize((640, 480), Image.ANTIALIAS) #640, 360
            imResize.save(output + file_name, 'JPEG', quality=90)
    print('Complete renaming and resizing')

resize(path, output)
