# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
import os

#tutorial 6 of 21
#img = cv2.imread('bookpage.jpg')
#greyscaled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#retval, threshold = cv2.threshold(img, 12, 255, cv2.THRESH_BINARY)
#gaus = cv2.adaptiveThreshold(greyscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
#cv2.imshow('original', img)
#cv2.imshow('threshold', threshold)
#cv2.imshow('gaus', gaus)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


#tutorial 7,8,9 of 21
# =============================================================================
# cap = cv2.VideoCapture(0)
# 
# while True:
#     _, frame = cap.read()
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     
#     # hsv hue sat value
#     #light blue colors sort of?
#     lower = np.array([100,40,120])
#     upper = np.array([150,100,200])
#     
#     #red?
#     #lower = np.array([130,120,120])
#     #upper = np.array([180,255,255])
#    
#     #orange?
#     #lower = np.array([10,128,128])
#     #upper = np.array([20,255,255])
#     
#     #light green?
#     #lower = np.array([46,120,120])
#     #upper = np.array([100,255,255])
#     
#     mask = cv2.inRange(hsv, lower, upper)
#     res = cv2.bitwise_and(frame, frame, mask = mask)
#     
#     kernel = np.ones((5,5), np.uint8)
#     opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#     closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#     
#    # blur = cv2.GaussianBlur(res, (15,15), 0)
#     #median = cv2.medianBlur(res, 15)
#     
#     cv2.imshow('frame', frame)
#    # cv2.imshow('mask', mask)
#     cv2.imshow('res', res)
#    # cv2.imshow('blur', blur)
#    # cv2.imshow('median', median)
#    # cv2.imshow('opening', opening)
#     #cv2.imshow('closing', closing)
#     
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#        break
# 
# cv2.destroyAllWindows()
# cap.release()
# =============================================================================


#tutorial 10 of 21
# =============================================================================
# cap = cv2.VideoCapture(0)
# 
# while True:
#     _, frame = cap.read()
#     
#     edges = cv2.Canny(frame, 100, 100)
#     
#     cv2.imshow('edges', edges)
#     
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#     
# cv2.destroyAllWindows()
# cap.release()
# =============================================================================



#tutorial 16 out of 21
# =============================================================================
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
# smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
# cap = cv2.VideoCapture(0)
# 
# while True:
#     ret, img = cap.read()
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)
# 
#     for (x,y,w,h) in faces:
#         cv2.rectangle(img, (x,y), (x+w-2, y+h+50), (255,0,0), 2)
#         roi_gray = gray[y:y+h, x:x+w]
#         roi_color = img[y:y+h, x:x+w]
#         eyes = eye_cascade.detectMultiScale(roi_gray, 1.2)
#        # smile = smile_cascade.detectMultiScale(roi_gray, 3.5)
#         for (ex,ey,ew,eh) in eyes:
#             cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)
#       #  for (sx,sy,sw,sh) in smile:
#        #     cv2.rectangle(roi_color, (sx,sy), (sx+sw, sy+sh), (0,0,255), 2) 
#     
#     #testing if statement based on camera visual
#     if len(faces) > 1:
#         edges = cv2.Canny(img, 100, 100)
#         cv2.imshow('edges', edges)
#     
#     cv2.imshow('img', img)
#     
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#     
# cv2.destroyAllWindows()
# cap.release()
# 
# =============================================================================


#tutorial 1 of 21
#img = cv2.imread('key.jpg', cv2.IMREAD_GRAYSCALE)
#cv2.imshow('image', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()






# cap = cv2.VideoCapture(0)

#while True:
 #   ret, frame = cap.read()
  #  cv2.imshow('frame', frame)
    
   # if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break
    
#cap.release()
#cv2.destroyAllWindows()

e1 = cv2.getTickCount()

# Load two images
img1 = cv2.imread('key.jpg')
img2 = cv2.imread('opencv-logo2.png')

#print(img2.shape)
# I want to put logo on top-left corner, So I create a ROI
rows,cols,channels = img2.shape
roi = img1[0:rows, 0:cols ]
#print(" rows of img2 " , rows , " cols of img2 " , cols , " channels of img2 " , channels)
# Now create a mask of logo and create its inverse mask also
img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 150, 255, cv2.THRESH_BINARY)
#print("ret is " , ret)
#print("mask is " , mask)
mask_inv = cv2.bitwise_not(mask)

# Now black-out the area of logo in ROI
img1_bg = cv2.bitwise_and(roi,roi,mask = mask)

# Take only region of logo from logo image.
img2_fg = cv2.bitwise_and(img2,img2,mask = mask_inv)

# Put logo in ROI and modify the main image
dst = cv2.add(img1_bg,img2_fg)
img1[0:rows, 0:cols ] = dst

cv2.imshow('img2gray', img2gray)
cv2.imshow('?', mask)
cv2.imshow('what', mask_inv)
cv2.imshow('chuck', img1_bg)
cv2.imshow('ok', img2_fg)
cv2.imshow('res',img1)
e2 = cv2.getTickCount()
t = (e2 - e1)/cv2.getTickFrequency()
print(t)
cv2.waitKey(0)
cv2.destroyAllWindows()

