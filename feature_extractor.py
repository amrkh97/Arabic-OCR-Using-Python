import cv2
import imutils
import numpy as np
from scipy import stats
from imutils import contours

###################Common Functions#################

class SeaparationRegion:
    EndIndex:int = 0
    StartIndex:int = 0
    CutIndex:int = 0
    
    def setEndIndex(self, i):
        self.EndIndex = i

    def setStartIndex(self, i):
        self.StartIndex = i

    def setCutIndex(self, i):
        self.CutIndex = i

def open(line):
    line_copy = np.copy(line)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    LineImage = cv2.morphologyEx(line_copy, cv2.MORPH_OPEN, kernel)
    return LineImage

def returnToBGR(image):
    return cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)

def find_nearest(VP, i):
    VP = np.asarray(VP)
    k = (np.abs(VP - i)).argmin()
    return VP[k]

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
    MAX = max(PV)
    
    for i in range(len(HP)):
        if HP[i] == MAX:
            BLI = i
            
    return BLI

def FindingMaximumTransitions(line_img, BLI): #5
    MaxTransitions = 0
    MTI = BLI
    i = BLI
    while i < line_img.shape[0]:
        CurrentTransitions = 0
        FLAG = 0
        j = 0
        while j < line_img.shape[1]:
            if line_img[i,j]==1 and FLAG == 0:
                CurrentTransitions += 1
                FLAG = 1
            elif line_img[i,j]!=1 and FLAG == 1:
                FLAG = 0
            j += 1
        #END WHILE
        if CurrentTransitions >= MaxTransitions:
             MaxTransitions = CurrentTransitions
             MTI = i
        i += 1
    #END WHILE
    MTI = line_img.shape[0] - MTI    
    return MTI

#WIP:
def CutPointIdentification(line, word, MTI): #6
    i = 1
    FLAG = 0
    LineImage = open(line)
    VP = getVerticalProjection(line)
    MFV = stats.mode(VP.tolist())[0][0]
    SeaparationRegions = []
    # print(word)
    while i <= word.shape[1] - 1:
        # Line 8
        if word[MTI, i] == 1 and FLAG == 0:
            SR = SeaparationRegion()
            SR.setEndIndex(i)
            FLAG = 1
        # Line 12
        elif word[MTI, i] != 1 and FLAG == 1:
            SR.setStartIndex(i)
            MidIndex = int((SR.EndIndex + SR.StartIndex)/2)
            VP = np.array(VP)

            k_equal_zero = np.where(VP[SR.StartIndex:SR.EndIndex] == 0)
            k_equal_zero = k_equal_zero[0]
            
            k_less_than_MFV_EndIndex = np.where(VP <= MFV) and np.where(VP < SR.EndIndex)
            k_less_than_MFV_EndIndex = k_less_than_MFV_EndIndex[0]

            k_less_than_MFV = np.where(VP[SR.StartIndex:SR.EndIndex] <= MFV)
            k_less_than_MFV = k_less_than_MFV[0]
            # k is an array of indices
            
            # Line 15
            if len(k_equal_zero) != 0:
                SR.setCutIndex(find_nearest(k_equal_zero, MidIndex))
            elif VP[MidIndex] == MFV:
                SR.setCutIndex(MidIndex)
            # Line 20
            elif len(k_less_than_MFV_EndIndex) != 0:
                SR.setCutIndex(find_nearest(k_less_than_MFV_EndIndex, MidIndex))
            # Line 23
            elif len(k_less_than_MFV) != 0:
                SR.setCutIndex(find_nearest(k_less_than_MFV, MidIndex))
            else:
                SR.setCutIndex(MidIndex)
            SeaparationRegions.append(SR)
            FLAG = 0
        i += 1
    return SeaparationRegions