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


while True:
    success, img = cap.read()
    imgOut = segmentor.removeBG(img, (255, 0, 255))
    cv2.imshow("Image", img)
    cv2.imshow("Image Out", imgOut)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
cv2.destroyAllWindows()