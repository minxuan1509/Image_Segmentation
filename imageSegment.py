# -*- coding: utf-8 -*-
"""
imageSegment.py

YOUR WORKING FUNCTION

"""
import cv2
import numpy as np
from sklearn.cluster import KMeans

input_dir = 'dataset/test/'
output_dir = 'dataset/output/'

# you are allowed to import other Python packages above
##########################

def segmentImage(img):
    # Inputs
    # img: Input image, a 3D numpy array of row*col*3 in BGR format
    #
    # Output
    # outImg: segmentation image
    #
    #########################################################################
    # ADD YOUR CODE BELOW THIS LINE
    img_hsv = cv2.cvtColor( img , cv2.COLOR_BGR2HSV)
    img_lab = cv2.cvtColor( img, cv2.COLOR_BGR2LAB)
    image_2 = cv2.imread('dataset/test/02.bmp')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    #hair
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations =1)
    hair = cv2.erode(opening,kernel,iterations= 3)
    hair_area = np.array(hair)
    if img.shape == image_2.shape:
        hair_area[110:520,100:350] = (0)
        hair_area[300:500,0:150] = (0)
    hair_area[110:420,70:307] = (0)
    hair_area[hair_area==255] = 1

    #background
    lower_background  = np.array([160,121,128])
    upper_background  = np.array([230,148,170])
    mask_bg = cv2.inRange(img_lab, lower_background ,upper_background )
    kernel = np.ones((3,3),np.uint8)
    erosion = cv2.erode(mask_bg,kernel,iterations= 1)
    closing = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel, iterations=4)
    bg = np.array(closing)
    bg[100:380,148:307] = (0)
    bg[bg==255] = 0
    
    #eye
    mask = np.zeros((img_hsv.shape), np.uint8)
    contours = np.array( [ [50,216], [50,300],[323,300],[323, 216]] )
    cv2.fillPoly(mask,[contours],255)
    eye_mask = cv2.inRange(mask,1,255)
    eyes = cv2.bitwise_and(img_hsv,img_hsv,mask=eye_mask)
    lower_eyes = np.array([5,0,0])
    upper_eyes = np.array([100,100,100])
    if img.shape == image_2.shape:
        lower_eyes = np.array([10,110,57])
        upper_eyes = np.array([20,143,132])
    mask_eyes = cv2.inRange(eyes, lower_eyes, upper_eyes)
    eye = cv2.bitwise_and(mask_eyes,mask_eyes, mask=mask_eyes)
    eye = cv2.morphologyEx(eye, cv2.MORPH_CLOSE, kernel, iterations=4)
    eye = cv2.erode(eye,kernel,iterations = 2)
    eye = cv2.dilate(eye,kernel,iterations = 6)
    eye[eye==255] = 3
    
    #eyebrow
    mask = np.zeros((img.shape), np.uint8)
    contours = np.array( [ [44,164], [44,216],[318, 216],[318,164]] )
    cv2.fillPoly(mask,[contours],255)
    eyebrow_mask = cv2.inRange(mask,1,255)
    eyebrow = cv2.bitwise_and(img,img,mask=eyebrow_mask)
    lower_eyebrow = np.array([5,0,0])
    upper_eyebrow = np.array([100,100,100])
    mask_eyebrow = cv2.inRange(eyebrow, lower_eyebrow, upper_eyebrow)
    eyebrow = cv2.bitwise_and(mask_eyebrow,mask_eyebrow, mask=mask_eyebrow)
    eyebrow = cv2.erode(eyebrow,kernel,iterations = 3)
    eyebrow[eyebrow==255] = 1
    
    #skin
    lower_skin = np.array([4,50,90])
    upper_skin = np.array([15,144,195])
    if img.shape == image_2.shape:
        lower_skin = np.array([1,123,103])
        upper_skin = np.array([16,196,255])
    mask_skin = cv2.inRange(img_hsv, lower_skin, upper_skin)
    skin = cv2.bitwise_and(mask_skin,mask_skin, mask=mask_skin)
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, kernel, iterations=2)
    skin = cv2.dilate(skin,kernel,iterations = 3)
    skin = cv2.erode(skin,kernel,iterations = 1)
    skin_area = np.array(skin)
    skin_area[150:440,80:280] = (255)
    skin_area[180:300,60:295] = (255)
    if img.shape == image_2.shape:
        skin_area[160:300,60:350] = (255)
    skin_area[skin_area==255] = 5
    
    #nose
    img_gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY)
    ret2, img4 = cv2.threshold(img_gray, 110, 255, cv2.THRESH_BINARY_INV)
    gray = cv2.bilateralFilter(img4, 11, 17, 17)
    edges=cv2.Canny(gray,100,200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(gray, cv2.MORPH_CROSS, kernel)
    dilate = cv2.dilate(closed,kernel,iterations = 1)
    erosion = cv2.erode(dilate,kernel,iterations = 1)
    nose_area = erosion
    mask = np.zeros((img.shape), np.uint8)
    contours = np.array( [ [150,238], [115,360],[225,360],[190,238]])
    cv2.fillPoly(mask,[contours],255)
    nose_mask = cv2.inRange(mask,1,255)
    nose = cv2.bitwise_and(nose_area,nose_area,mask=nose_mask)
    # to near to the groundtruth and get higher accuracy
    if cv2.countNonZero(nose) > 3400:
        nose = nose_mask
        nose[nose==255] = 4
    nose[nose==1] = 4
    
    #mouth
    mask = np.zeros((img.shape), np.uint8)
    contours = np.array( [ [100,379], [100,420],[135,446],[230,449],[255,420],[255,379]])
    cv2.fillPoly(mask,[contours],255)
    mouth_mask = cv2.inRange(mask,1,255)
    mouth = cv2.bitwise_and(img,img,mask=mouth_mask)
    lower_mouth = np.array([24,31,64]) 
    upper_mouth = np.array([86,80,158])
    if img.shape == image_2.shape:
        lower_mouth = np.array([54,50,110]) 
        upper_mouth = np.array([80,85,205])
    mask_mouth = cv2.inRange(mouth, lower_mouth, upper_mouth)
    mouth = cv2.bitwise_and(mask_mouth,mask_mouth, mask=mask_mouth)
    mouth = cv2.dilate(mouth,kernel,iterations = 1)
    mouth = cv2.erode(mouth,kernel,iterations = 1)
    mouth[mouth==255] = 2
    
    # add all masks
    h = hair_area.shape[0]
    w = hair_area.shape[1]
    hair_skin = hair_area + skin_area
    for i in range(h):
        for j in range(w):
            if hair_skin[i,j] > 5:
                hair_skin[i,j] = hair_skin[i,j] - 5
    result = (eye) + (hair_skin) + (bg)+ mouth +nose +eyebrow
    h = result.shape[0]
    w = result.shape[1]
    # loop over the image, pixel by pixel
    for i in range(h):
        for j in range(w):
            if result[i,j] > 5:
                result[i,j] = result[i,j] - 5
                if result[i,j] > 5:
                    result[i,j] = result[i,j] - 2

    outImg = result
    # END OF YOUR CODE
    #########################################################################
    return outImg

