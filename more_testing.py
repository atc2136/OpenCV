# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 19:38:41 2018

@author: Nike
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

# This example from image pyramids doc isn't working
# =============================================================================
# A = cv2.imread('Apple.png')
# B = cv2.imread('Orange.png')
# 
# # generate Gaussian pyramid for A
# G = A.copy()
# gpA = [G]
# for i in range(6):
#     G = cv2.pyrDown(G)
#     gpA.append(G)
# 
# # generate Gaussian pyramid for B
# G = B.copy()
# gpB = [G]
# for i in range(6):
#     G = cv2.pyrDown(G)
#     gpB.append(G)
# 
# # generate Laplacian Pyramid for A
# lpA = [gpA[5]]
# for i in range(5,0,-1):
#     GE = cv2.pyrUp(gpA[i])
#     L = cv2.subtract(gpA[i-1],GE)
#     lpA.append(L)
# 
# # generate Laplacian Pyramid for B
# lpB = [gpB[5]]
# for i in range(5,0,-1):
#     GE = cv2.pyrUp(gpB[i])
#     L = cv2.subtract(gpB[i-1],GE)
#     lpB.append(L)
# 
# # Now add left and right halves of images in each level
# LS = []
# for la,lb in zip(lpA,lpB):
#     rows,cols,dpt = la.shape
#     ls = np.hstack((la[:,0:cols/2], lb[:,cols/2:])) #Not sure what this means
#     LS.append(ls)
# 
# # now reconstruct
# ls_ = LS[0]
# for i in range(1,6):
#     ls_ = cv2.pyrUp(ls_)
#     ls_ = cv2.add(ls_, LS[i])
# 
# # image with direct connecting each half
# real = np.hstack((A[:,:cols/2],B[:,cols/2:]))
# 
# cv2.imwrite('Pyramid_blending2.jpg',ls_)
# cv2.imwrite('Direct_blending.jpg',real)
# =============================================================================



# =============================================================================
# #Testing Pyramids
# img = cv2.imread('opencv-logo2.png')
# lower_reso = cv2.pyrDown(img)
# lower_reso2 = cv2.pyrDown(lower_reso)
# lower_reso3 = cv2.pyrDown(lower_reso2)
# lower_reso4 = cv2.pyrDown(lower_reso3)
# 
# plt.subplot(1,5,1), plt.imshow(img, cmap = 'gray')
# plt.title('Original'), plt.xticks([]), plt.yticks([])
# plt.subplot(1,5,2), plt.imshow(lower_reso, cmap = 'gray')
# plt.title('1'), plt.xticks([]), plt.yticks([])
# plt.subplot(1,5,3), plt.imshow(lower_reso2, cmap = 'gray')
# plt.title('2'), plt.xticks([]), plt.yticks([])
# plt.subplot(1,5,4), plt.imshow(lower_reso3, cmap = 'gray')
# plt.title('1'), plt.xticks([]), plt.yticks([])
# plt.subplot(1,5,5), plt.imshow(lower_reso4, cmap = 'gray')
# plt.title('2'), plt.xticks([]), plt.yticks([])
# =============================================================================

# =============================================================================
# #Edge Detection
# img = cv2.imread('opencv-logo2.png')
# edges = cv2.Canny(img, 100, 200)
# 
# plt.subplot(1,2,1), plt.imshow(img, cmap = 'gray')
# plt.title('Original'), plt.xticks([]), plt.yticks([])
# plt.subplot(1,2,2), plt.imshow(edges, cmap = 'gray')
# plt.title('Canny'), plt.xticks([]), plt.yticks([])
# 
# plt.show()
# =============================================================================

# =============================================================================
# # Morphological operations
# img = cv2.imread('j.png')
# kernel = np.ones((5,5), np.uint8)
# erosion = cv2.erode(img, kernel, iterations = 1)
# dilation = cv2.dilate(img, kernel, iterations = 1)
# opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
# closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
# gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
# tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
# blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
# #erode_dilate = cv2.dilate(erosion, kernel, iterations = 1)
# 
# 
# cv2.imshow('original', img)
# cv2.imshow('erosion', erosion)
# cv2.imshow('dilation', dilation)
# cv2.imshow('opening', opening)
# cv2.imshow('closing', closing)
# cv2.imshow('gradient', gradient)
# cv2.imshow('tophat', tophat)
# cv2.imshow('blackhat', blackhat)
# #cv2.imshow('combo', erode_dilate)
# 
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# =============================================================================

# =============================================================================
# Blurring
# img = cv2.imread('noisy_balloon.jpg')
# median = cv2.medianBlur(img, 5)
# blur = cv2.GaussianBlur(img, (5,5), 0)
# bilateral = cv2.bilateralFilter(img, 9, 75, 75)
# 
# cv2.imshow('pic', img)
# cv2.imshow('median', median)
# cv2.imshow('blur', blur)
# cv2.imshow('bilateral', bilateral)
# 
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# =============================================================================


#flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
#print(flags)

# =============================================================================
# img = cv2.imread('opencv-logo2.png')
# 
# kernel = np.ones((5,5), np.float32)/25
# dst = cv2.filter2D(img, -1, kernel)
# 
# plt.subplot(121), plt.imshow(img), plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(dst), plt.title('Averaging')
# plt.xticks([]), plt.yticks([])
# plt.show()
# 
# =============================================================================
#practice extracting more than one color at a time in video feed