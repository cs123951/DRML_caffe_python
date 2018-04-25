# -*- coding: utf-8 -*-
import cv2
import os
import numpy as np
from PIL import Image
import random
disfa_root = "F:/dataset/DISFA/ActionUnit_Labels"
people_num = [1,2,3,4,5,6,7,8,9,10,11,12,13,16,17,18,21,23,24,25,26,27,28,29,30,31,32]
au_num =[1,2,4,5,6,9,12,15,17,20,25,26]
#write to txt
def write_label():
    disfa_root = "F:/dataset/DISFA/ActionUnit_Labels"
    people_num = [1,2,3,4,5,6,7,8,9,10,11,12,13,16,17,18,21,23,24,25,26,27,28,29,30,31,32]
    au_num =[1,2,4,5,6,9,12,15,17,20,25,26]
    for vd_id in people_num:
        people_au = np.zeros([4844,12])
        au_root = disfa_root + "/SN" + str(vd_id).zfill(3)
        for au_index in range(len(au_num)):
            au_id = au_num[au_index]
            f = open(au_root+"/SN"+str(vd_id).zfill(3)+"_au"+str(au_id)+".txt")
            lines = f.readlines()
            for i in range(4844):
                line = lines[i]
                line = line.strip("\n")
                au_label = int(line.split(",")[1])
                people_au[i,au_index] = au_label
                if au_label > 1:
                    people_au[i,au_index] = 1
                else:
                    people_au[i,au_index] = 0
            f.close()
        f=open('F:/dataset/DISFA/AU/'+str(vd_id)+'.txt','a')
        for i in range(4844):
            for j in range(len(au_num)):
                f.write(str(int(people_au[i,j]))+" ")
            f.write("\n")
        f.close()        

def detectFaces(image_name):
    img = cv2.imread(image_name)
    face_cascade = cv2.CascadeClassifier("E:/Projects/jupyterProject/Face/disfa/haarcascade_frontalface_default.xml")
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img #if语句：如果img维度为3，说明不是灰度图，先转化为灰度图gray，如果不为3，也就是2，原图就是灰度图

    faces = face_cascade.detectMultiScale(gray, 1.2, 5)#1.3和5是特征的最小、最大检测窗口，它改变检测结果也会改变
    result = []
    for (x,y,width,height) in faces:
#        center_x = x + width//2
#        center_y = y + height//2
#        result.append((center_x-150,center_y-150,center_x+150,center_y+150))
        result.append((x,y,x+width,y+height))
    return result

def saveFaces(image_name,save_dir):
    faces = detectFaces(image_name)
    if faces:
        #将人脸保存在save_dir目录下。
        #Image模块：Image.open获取图像句柄，crop剪切图像(剪切的区域就是detectFaces返回的坐标)，save保存。
        pic_name = image_name.split('/')[-1]
#        save_dir = save_dir.lstrip('F:\\')
        
#         os.mkdir(save_dir)
#        file_name=save_dir.rstrip(save_dir.split('\\')[5])
#        if not os.path.exists(file_name):
#            os.makedirs(file_name)
        count = 0
        for (x1,y1,x2,y2) in faces:
            image_crop_name = save_dir+"\\"+pic_name
            Image.open(image_name).crop((x1,y1,x2,y2)).resize((170,170),Image.ANTIALIAS).save(image_crop_name)
            count+=1
        if count > 1:
            f=open("F:\\multiImage.txt","a")
            f.write(image_name)
            f.close()
    else:
        f=open("F:\\noImage.txt","a")
        f.write(image_name)
        f.close()

#detect images
def Crop_image():
    pic_root = "F:/dataset/DISFA/Pictures_Right"
    save_dir = "F:/dataset/DISFA/Crop_Right"
#    people_num = [1,2,3,4,5,6,7,8,9,10,11,12,13,16,17,18,21,23,24,25,26,27,28,29,30,31,32]
    for people_id in range(3,28):
        for seq_id in range(1,4845):
            image_path = pic_root+"/SN"+str(people_id).zfill(3)+"_"+str(seq_id).zfill(4)+".png"
            saveFaces(image_path,save_dir)
        print(str(people_id)+" done")
            
def Write_to_txt():
    people_num = [1,2,3,4,5,6,7,8,9,10,11,12,13,16,17,18,21,23,24,25,26,27,28,29,30,31,32]
    ff = open('F:/dataset/DISFA/AU_labels.txt','a')
    for vd_index in range(len(people_num)):
        vd_id = people_num[vd_index]
        f=open('F:/dataset/DISFA/AU/'+str(vd_id)+'.txt','a')
        lines = f.readlines()
        for i in range(4844):
            ff.write("F:/dataset/DISFA/Crop_Right/SN"+str(vd_index).zfill(3)+"_"+str(i).zfill(4)+".png ")
            ff.write(lines[i])
        f.close()
    ff.close()
#write_label()
#Crop_image()
#Write_to_txt()

def write_image_label():
    txt_path = "F:/dataset/DISFA/AU"
    pic_path = "F:/dataset/DISFA/Crop_Right/"
    people = os.listdir(txt_path)
    f = open("pic_label.txt","a")
    for pp in people:
        ff = open(txt_path+"/"+pp)
        people_index = pp.split(".")[0]
        seq = ff.readlines()
        for i in range(4844):
            pic_name = "SN"+str(people_index.zfill(3))+"_"+str(str(i).zfill(4))+".png"
            if os.path.exists(pic_path+pic_name):
                f.write(pic_path+pic_name+" ")
                f.write(seq[i])
        ff.close()
    f.close()
#write_image_label()
def shuffle_data():
    f = open("pic_label.txt")
    lines = f.readlines()
    f.close()
    random.shuffle(lines)
    ff = open("disfa_data_label.txt","a")
    for i in lines:
        ff.write(i)
    ff.close()
shuffle_data()
    
    
    