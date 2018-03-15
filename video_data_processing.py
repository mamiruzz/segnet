# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 08:29:51 2018

@author: mamiruzz
"""


import cv2
#import datetime
import os

#video file location
video_file_location = '1091516-L-1.MP4'
vidcap = cv2.VideoCapture(video_file_location) 


#make sure to create a folder to save those frames

frame_ext = '.jpg'
#print(os.path.basename(video_file_location))
dir_loc =os.path.splitext(video_file_location)[0]
vid_name = os.path.basename(dir_loc)
print(vid_name)

save_location = './'+str(vid_name)+'/'
if not os.path.exists(save_location):
    os.makedirs(save_location)

# image is an array of array of [R,G,B] values
success,image = vidcap.read()


#1000 = 1 seconds interval
interval = 1000 
rate = interval

csv_file = vid_name+'.csv'

if os.path.isfile(csv_file)!=True:
        with open(csv_file, 'a') as f:
            f.write('fileName, frameName\n')

count = 0; 
while success:
    # just cue to Interval sec. position
    vidcap.set(cv2.CAP_PROP_POS_MSEC, interval)      
    success,image = vidcap.read()
    #file_time = str(datetime.timedelta(seconds=interval/1000)) 
    #print(file_time)
    
    # save frame as JPEG file
    #frame_number = count+(rate/1000)
    #print(frame_number)
    print(int(count))
    frame_name = "{}-{}".format(vid_name, str(int(count)))
    frame_location ="{}{}{}".format(save_location, frame_name, frame_ext)
    
    # save frame as JPEG file
    resize = cv2.resize(image, (640, 480)) 
    cv2.imwrite(frame_location, resize)
    with open(csv_file, 'a') as f:
        f.write('{}, {}\n'.format(vid_name, frame_name))
    # exit if Escape is hit
    if cv2.waitKey(10) == 27:
        break
    interval+=rate
    count += (rate/1000)
print('framing complete!!')