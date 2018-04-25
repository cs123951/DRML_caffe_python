# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 17:05:25 2017

@author: Admin
"""

import cv2
import os
video_root = "F:/dataset/DISFA/Video_RightCamera"
videos = os.listdir(video_root)
#for vd_id in range(len(videos)):
#    file_dir = "F:/dataset/DISFA/Pictures_Right/SN" + str(vd_id+1).zfill(3)
#    if not os.path.exists(file_dir):
#        os.mkdir(file_dir)
for vd_id in range(len(videos)):
    video_path = video_root + "/" + videos[vd_id]
    vc = cv2.VideoCapture(video_path) #读入视频文件
    if vc.isOpened(): #判断是否正常打开
        rval , frame = vc.read()
    else:
        rval = False
    c=1
    while rval:   #循环读取视频帧
        cv2.imwrite('F:/dataset/DISFA/Pictures_Right/SN'+str(vd_id+1).zfill(3) +'_'+str(c).zfill(4)+ '.png',frame) #存储为图像
        rval, frame = vc.read()
        c = c + 1
    vc.release()




#ffmpeg -y -ss 0 -i RightVideoSN001_Comp.avi -f image2  -r 1 -t 8 -qscale 1 image/%5d.jpg