import cv2
import imutils
import numpy as np
from scipy import stats
from imutils import contours
from commonfunctions import *

def returnToBGR(image):
    return cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)

def correct_skew(thresh):
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
	    angle = -(90 + angle)
    else:
	    angle = -angle
    
    (h, w) = thresh.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def getVerticalProjection(img):
    return (np.sum(img, axis = 0))//255

def getHorizontalProjection(img):
    return (np.sum(img, axis = 1))//255

####################################################

def DetectLines(line_img):
    input_image = correct_skew(line_img) 
    _,thresh = cv2.threshold(input_image, 0, 255,cv2.THRESH_OTSU|cv2.THRESH_BINARY_INV)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,5))
    dilation = cv2.dilate(thresh, rect_kernel, iterations = 1)
    
    contours_initial, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    im2 = input_image.copy()
    
    Detected_Lines = []
    # sort the contours
    for method in ("right-to-left","top-to-bottom"):
	    contours_initial, boundingBoxes = contours.sort_contours(contours_initial, method=method)
    for cnt in contours_initial:
            x, y, w, h = cv2.boundingRect(cnt)
            fx = x+w
            fy = y+h
            cv2.rectangle(im2, (x, y), (fx, fy), (100, 100, 100), 2)
            trial_image = input_image[y:fy,x:fx]
            Detected_Lines.append(trial_image)
    
    return Detected_Lines

def DetectWords(line_img):
    input_image = correct_skew(line_img) 
    _,thresh = cv2.threshold(input_image, 127, 255,cv2.THRESH_OTSU|cv2.THRESH_BINARY_INV)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilation = cv2.dilate(thresh, rect_kernel, iterations = 1)
    contours_initial, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    Detected_Words = []
    # sort the contours
    for method in ("right-to-left"):
	    contours_initial, boundingBoxes = contours.sort_contours(contours_initial, method=method)
    for cnt in contours_initial:
            x, y, w, h = cv2.boundingRect(cnt)
            fx = x+w
            fy = y+h
            trial_image = input_image[y:fy,x:fx]
            Detected_Words.append(trial_image)
    return Detected_Words

def BaselineDetection(line_img): #4
    PV = []
    BLI = 0
    _,thresh_img = cv2.threshold(line_img,127,255,cv2.THRESH_BINARY_INV)
    thresh_img = np.asarray(thresh_img)
    
    HP = getHorizontalProjection(thresh_img)
    PV_Indices = (HP > np.roll(HP,1)) & (HP > np.roll(HP,-1))
    
    for i in range(len(PV_Indices)):
        if PV_Indices[i] == True:
            PV.append(HP[i])
    #Temp Fix:
    if len(PV) != 0:
        MAX = max(PV)
    else:
        MAX = 36
    
    for i in range(len(HP)):
        if HP[i] == MAX:
            BLI = i
    #BLI = line_img.shape[0] - BLI        
    return BLI


####################################################################

def amrsFunction(word,BLI):
    VP = getVerticalProjection(word)
    copyWord = np.copy(word)
    VP[VP < 2] = 0
    copyWord[BLI, VP == 0] = 0
    return copyWord

def extractLettersFromWord(word):
    VP = getVerticalProjection(word)
    VPIndices = np.where(VP == 0)[0]
    temp = np.copy(VPIndices)
    temp = np.diff(temp)
    temp = np.insert(temp,0,0)
    VPIndices = VPIndices[temp > 1]
    
    return VPIndices

def showImagesFromSegments(word,listOfSegments):
    i = 0
    finalSegments = []
    for segment in listOfSegments:
        finalSegments.append(word[:,i:segment])
        i = segment
    return finalSegments

def extractFromWord(word,BLI):
    word = threshold(word)
    word = word * 255          
    wordCopy = amrsFunction(word,BLI)
    listOfSegmentations = extractLettersFromWord(wordCopy)
    return showImagesFromSegments(word,listOfSegmentations)

def preprocessIntoWords(input_image):
    if len(input_image.shape) == 3:
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    input_image = correct_skew(input_image)
    return DetectLines(input_image)


def extractSeparateLettersWholeImage(input_image):
    len_words = preprocessIntoWords(input_image)
    all_words = []
    for line in len_words:
    
        BLI = BaselineDetection(line)
        Detected_Words = np.flip(DetectWords(line))
    
        for word in Detected_Words:  
            SegmentedWord = extractFromWord(word,BLI)
            #TODO: Add filteration to remove small segments that are irrelevant.
            all_words.append(SegmentedWord)
    return all_words   
