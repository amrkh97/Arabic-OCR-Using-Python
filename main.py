import numpy as np
import glob
import skimage.io as io
from skimage import util
from skimage.color import rgb2gray
from feature_extractor import extract
import cv2

'''
Images_Path = './Test Data Set/'
Text_Path = './Test Data Set/'
'''

Images_Path = './Pattern Data Set/scanned/'


Number_Of_Files = 1 #Sample of Files to check on

Actual = 0
Detected = 0
gen =  glob.iglob(Images_Path + "*.png")
for i in range(Number_Of_Files):
    py = next(gen)
    input_image = io.imread(py)
    #input_image = np.array(input_image, dtype=np.uint8)
    if len(input_image.shape) == 3:
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    ret,thresh1 = cv2.threshold(input_image, 0, 255,cv2.THRESH_OTSU|cv2.THRESH_BINARY_INV)
    cv2.imshow('thresh1', thresh1)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)
    cv2.imshow('dilation', dilation)

    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    im2 = input_image.copy()
    for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow('final', im2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
