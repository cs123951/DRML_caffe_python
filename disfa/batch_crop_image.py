# -*- coding: utf-8 -*-
import cv2
import os
from PIL import Image

import scipy.misc as misc

def detectFaces(image_name):
    img = cv2.imread(image_name)
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img #if语句：如果img维度为3，说明不是灰度图，先转化为灰度图gray，如果不为3，也就是2，原图就是灰度图

    faces = face_cascade.detectMultiScale(gray, 1.2, 5)#1.3和5是特征的最小、最大检测窗口，它改变检测结果也会改变
    result = []
    for (x,y,width,height) in faces:
        center_x = x + width//2
        center_y = y + height//2
        result.append((center_x-150,center_y-150,center_x+150,center_y+150))
#        result.append((x,y,x+width,y+height))
    return result

def saveFaces(image_name):
    faces = detectFaces(image_name)
    if faces:
        #将人脸保存在save_dir目录下。
        #Image模块：Image.open获取图像句柄，crop剪切图像(剪切的区域就是detectFaces返回的坐标)，save保存。
        save_dir = image_name.split('/')[-1].split('.')[0]
        save_dir = save_dir.lstrip('F:\\')
        save_dir = "F:\\crop_data\\" + save_dir
#         os.mkdir(save_dir)
        file_name=save_dir.rstrip(save_dir.split('\\')[5])
        if not os.path.exists(file_name):
            os.makedirs(file_name)
        count = 0
        for (x1,y1,x2,y2) in faces:
            image_crop_name = save_dir+".png"
            Image.open(image_name).crop((x1,y1,x2,y2)).resize((170,170),Image.ANTIALIAS).save(image_crop_name)
            count+=1
        if count > 1:
            f=open("F:\\multiImage.txt","w")
            f.write(image_name)
    else:
        f=open("F:\\noImage.txt","w")
        f.write(image_name)

#detect images
def ListFilesToTxt(dir,file,wildcard,recursion):
     exts = wildcard.split(" ")
     files = os.listdir(dir)
     for name in files:
         fullname=os.path.join(dir,name)
         if(os.path.isdir(fullname) & recursion):
             ListFilesToTxt(fullname,file,wildcard,recursion)
         else:
             for ext in exts:
                 if(name.endswith(ext)):
                     file.write(fullname + "\n")
                     break

def Test():
   dir="F:\\cohn-kanade-images"
   outfile="F:\\imageName.txt"
   wildcard = ".png"
   
   file = open(outfile,"w")
   if not file:
     print ("cannot open the file %s for writing" % outfile)

   ListFilesToTxt(dir,file,wildcard, 1)
   
   file.close()

Test()

            
# detect face and crop            
f = open("F:\\imageName.txt")
for image_name in f.readlines():
    saveFaces(image_name.strip('\n'))

#generate txt file


