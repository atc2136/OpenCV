# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 18:05:21 2018

@author: Nike
"""
import cv2
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
import os

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(imgray,50,255,0)
    img2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.drawContours(img, contours, -1, (0,255,0), 3)
    
    hierarchy = hierarchy[0] # get the actual inner list of hierarchy descriptions

# For each contour, find the bounding rectangle and draw it
# =============================================================================
#     for component in zip(contours, hierarchy):
#         currentContour = component[0]
#         currentHierarchy = component[1]
#         x,y,w,h = cv2.boundingRect(currentContour)
#         if currentHierarchy[2] < 0:
#             # these are the innermost child components
#             cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),3)
#         elif currentHierarchy[3] < 0:
#             # these are the outermost parent components
#             cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),3)
# =============================================================================
    

    cv2.imshow('img', img)
    cv2.imshow('test', thresh)
    cv2.imshow('img2', img2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()

# =============================================================================
# 
# img = cv2.imread('opencv-logo2.png')
#  
# imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(imgray,80,255,0)
# img2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# 
# cv2.drawContours(img, contours, -1, (0,255,255), 3)
# #for i in hierarchy:
#   
# print(contours)  
# print(hierarchy)
# 
# hierarchy = hierarchy[0] # get the actual inner list of hierarchy descriptions
# 
# # For each contour, find the bounding rectangle and draw it
# for component in zip(contours, hierarchy):
#     currentContour = component[0]
#     currentHierarchy = component[1]
#     x,y,w,h = cv2.boundingRect(currentContour)
#     if currentHierarchy[2] < 0:
#         # these are the innermost child components
#         cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),3)
#     elif currentHierarchy[3] < 0:
#         # these are the outermost parent components
#         cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),3)
# 
# #kernel = np.ones((5,5), np.float32)/25
# #dst = cv2.filter2D(img, -1, kernel)
#  
# cv2.imshow('img',img)
# 
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# =============================================================================
