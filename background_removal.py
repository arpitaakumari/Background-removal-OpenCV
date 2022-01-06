# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 19:20:37 2021

@author: ARPITA KUMARI
"""

import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
segmentor = SelfiSegmentation(1)
fpsReader = cvzone.FPS()

listImg = os.listdir("Bg Images")
imgList = []
for imgPath in listImg:
    img = cv2.imread(f'Bg Images/{imgPath}')
    imgList.append(img)
    
indexImg = 0

 
while True:
    success, img = cap.read()
    imgOut = segmentor.removeBG(img, imgList[indexImg], threshold= 0.85)
    
    imgStacked = cvzone.stackImages([img, imgOut], 2, 1)
    _, imgStacked = fpsReader.update(imgStacked, color = (0, 0, 255))
    
    cv2.imshow("Image", imgStacked)
    key = cv2.waitKey(1)
    if key == ord('a'):
        if indexImg==0:
            indexImg = len(imgList)-1
        else:
            indexImg -= 1
    elif key == ord('d'):
        if indexImg==len(imgList)-1:
            indexImg = 0
        else:
            indexImg += 1
    elif key == ord('q'):
        break
cv2.destroyAllWindows()