# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 08:29:51 2018

@author: mamiruzz
"""


import cv2
#import datetime
import os

#video file location
video_file_location = 'C:/Users/mamiruzz/Downloads/Videos/g_media_left.MP4'
vidcap = cv2.VideoCapture(video_file_location) 

save_location = './20sec/'
frame_ext = '.jpg'
#print(os.path.basename(video_file_location))
dir_loc =os.path.splitext(video_file_location)[0]
vid_name = os.path.basename(dir_loc)
print(vid_name)

# image is an array of array of [R,G,B] values
success,image = vidcap.read()


#5000 = 5 seconds interval
interval = 60000 
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
    frame_name = "{}-{}".format(vid_name, str(count+interval))
    frame_location ="{}{}{}".format(save_location, frame_name, frame_ext)
    
    # save frame as JPEG file
    cv2.imwrite(frame_location, image)
    with open(csv_file, 'a') as f:
        f.write('{}, {}\n'.format(vid_name, frame_name))
    # exit if Escape is hit
    if cv2.waitKey(10) == 27:
        break
    interval+=rate
    count += 1