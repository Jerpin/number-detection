#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 17:15:02 2020

@author: jerry
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 15:09:58 2020

@author: Jerry
"""

import cv2
import numpy as np
from skimage import io
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from scipy.ndimage.measurements import label
import time
from skimage import data,img_as_float
from skimage import data,img_as_ubyte
ipcam = cv2.VideoCapture(0)
model = tf.keras.models.load_model('model_CNN.h5')
while True:
    try:
        stat, image = ipcam.read()
        
        if cv2.waitKey(1) == 27:
            ipcam.release()
            cv2.destroyAllWindows()
            break
        #change the image into HSV tyoe for counting brightness
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_float = img_as_float(hsv)

        #counting brightness
        brightness = 0
        x = 0
        y = 0
        pix4 = 0
        for y in range(int(np.int64(hsv.shape[0])/4)):
            for x in range(int(np.int64(hsv.shape[1])/4)):
                brightness += hsv_float[y*4,x*4,2]
                pix4 += 1
        
        brightness = brightness/pix4*255
        print("brightness:",brightness)
        print(pix4)


        #Image processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresholding = 130 + int((brightness-60)/2)
        print('thresholding: ' , thresholding)
        blur1 = gaussian_filter(gray, sigma=1)
        blur = gaussian_filter(blur1, sigma=1)
        sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpen = cv2.filter2D(blur, -1, sharpen_kernel)
        thresh = cv2.threshold(sharpen,thresholding,255, cv2.THRESH_BINARY_INV)[1]
        cv2.imshow('thresh',thresh)
        thresh_inverse = 255-thresh


        #Capture the square area
        cnts = cv2.findContours(thresh_inverse, \
                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        min_area = 1500 #5000
        max_area = 20000
        images = 0
        img_NO = 0
        #setup a matrix with initial value that are all zeros
        imgs = np.full([128,128,2] , 0).astype(np.uint8)
        img28 = np.full([28,28,2] , 0).astype(np.uint8)

        for c in cnts:
            area = cv2.contourArea(c)
            if area > min_area and area < max_area:
                x,y,w,h = cv2.boundingRect(c)
                #check if the square fits the h/w ratio of a A4 paper
                if h/w<1.136 or h/w>1.764:
                    continue
                elif x==0 or x+w == image.shape[1]:
                    continue
                elif y == 0 or y+h == image.shape[0]:
                    continue
                else:
                    pass
                print('---------')
                print('area:' , area)
                print("x:" , x , x+w)
                print("y:" , y , y+h)
                ROI = thresh_inverse[y:y+h, x:x+w]
                kernel_dilate = np.ones((3,3),np.uint8)
                ROI = cv2.dilate(ROI , kernel_dilate , iterations = 2)
                ROI = cv2.erode(ROI , kernel_dilate , iterations = 1)
                ##### deleting areas that are not parts of papers
                y1 = int(ROI.shape[0]/7)
                y2 = int(ROI.shape[0]/7) * (-1)
                x1 = int(ROI.shape[1]/10)
                x2 = int(ROI.shape[1]/10)* (-1)
                ROI = ROI[y1:y2, x1:x2]
                ##### deleting areas that are not parts of papers
                pixel = []
                exist = []
                for x_ROI in range(ROI.shape[1]):
                    for y_ROI in range(ROI.shape[0]):
                        if ROI[y_ROI,x_ROI] == 0:
                            pixel.append(1)
                        else:
                            pixel.append(0)
                    if max(pixel) == 1 and min(pixel) == 0:
                        exist.append(1)
                    else:
                        exist.append(0)
                    pixel = []       
                steps = 0
                step_x = []
                for i in range(len(exist)-1):
                    if exist[i] != exist[i+1]:
                        steps += 1
                        step_x.append(i)
                    else:
                        pass
                if exist[0] == 1:
                    try:
                        ROI = ROI[ : , step_x[0]+1 : step_x[-1]-1]
                    except:
                        continue
                    
                #shapening
                ROI = cv2.threshold(ROI,110,255, cv2.THRESH_BINARY_INV)[1]
                #plot(ROI)
                ROI = cv2.resize(ROI,(128,128), interpolation = cv2.INTER_LINEAR) 
                cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
                
                #####Check how many numbers are contained in the square
                kernel_erode = np.ones((3,3),np.uint8)
                erosion = cv2.erode(ROI , kernel_erode , iterations = 1)
                erosion28 = cv2.resize(erosion,(28,28) , interpolation = cv2.INTER_LINEAR)
                #cv2.imwrite('ROI_{}.png'.format(images) , erosion28)
                images += 1
                #plot(erosion28)
                #if max(erosion28[:,0])!=0 or max(erosion28[:,-1])!=0 or\
                   #max(erosion28[0,:])!=0 or max(erosion28[-1:0])!=0:
                    #print(erosion28[:,-1])
                    #continue
                pixel = []
                exist = []
                for a in range(erosion28.shape[1]):
                    for b in range(erosion28.shape[0]):
                        if erosion28[b,a] == 0:
                            pixel.append(1)
                        else:
                            pixel.append(0)
                    if max(pixel) == 1 and min(pixel) == 0:
                        exist.append(1)
                    else:
                        exist.append(0)
                    pixel = []
                
                steps = 0
                
                for i in range(len(exist)-1):
                    if exist[i] != exist[i+1]:
                        steps += 1
                    else:
                        pass
                print(exist)
                numbers = steps/2 
                print("numbers contained in the paper:" , numbers)
                
                if numbers == 1.0:
                    #store the image into a matrix
                    imgs[:,:,img_NO] = erosion
                    img28[:,:,img_NO] = erosion28
                    img_NO += 1

                    cv2.imshow('e',erosion28)
                    img_exp = np.expand_dims(erosion28 , axis=0)
                    img_exp = np.expand_dims(img_exp , axis=3)
                    predict = model.predict_classes(img_exp)
                    print(predict)
                    if predict == [1]:
                        print('1success')
                        text = '1'
                        cv2.putText(image, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 255, 255), 1, cv2.LINE_AA)
                    elif predict == [3]:
                        print('3success')
                        text = '3'
                        cv2.putText(image, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 255, 255), 1, cv2.LINE_AA)
                    else:
                        text = '%s' %predict
                        cv2.putText(image, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 255, 255), 1, cv2.LINE_AA)
                    print(text)
                    

                else:
                    cv2.imshow('m',erosion28)
                    print("Mltiple numbers")
                    text = 'M'
                    cv2.putText(image, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 255), 1, cv2.LINE_AA)
                    pass #skip number 13
                 
    except:
        pass
    cv2.imshow('Image', image)
