import cv2
import imutils
import numpy as np
from scipy import stats
from imutils import contours
from commonfunctions import *
from skimage.measure import regionprops
import glob
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
    if BLI < word.shape[0]:
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
    privious_final_segment_index = -1
    for segment in listOfSegments:
        if  not (Is_excess_segment(word[:,i:segment])):
            finalSegments.append(word[:,i:segment])
            privious_final_segment_index += 1
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

def Is_excess_segment (char):
    HP = getHorizontalProjection (char)
    HP[HP !=0] = 1
    HPIndices = np.where(HP == 1)[0]
    HP_Indices_diff = np.diff(HPIndices)  
    if (np.sum(char)//255) <= 4 and not( np.any(HP_Indices_diff > 1)):
        return True
    return False

def extractSeparateLettersWholeImage(input_image):
    len_words = preprocessIntoWords(input_image)
    all_words = []
    for line in len_words:
        
        BLI = BaselineDetection(line)
        Detected_Words = DetectWords(line)
        
        Detected_Words.reverse()

        for word in Detected_Words:  
            SegmentedWord = extractFromWord(word,BLI)
            all_words.append(SegmentedWord)
    return all_words   

#####################
#Abdelgawad
def Gabor_filter (img):
    g_kernel = cv2.getGaborKernel((21, 21), 8.0, np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
    filtered_img = cv2.filter2D(img, cv2.CV_8UC3, g_kernel)
    filtered_img = np.array(filtered_img)
    features = filtered_img.flatten()
    return features

def horizontal_transitions(img):
    horizontal_transition_count = 0
    col_found_in = {}
    for r in range(img.shape[0]):
        for c in range(img.shape[1]-1):
            if img[r,c] != img[r,c+1] and  not( c in col_found_in.keys()) :
                col_found_in[c] = 1
                horizontal_transition_count+=1
    return horizontal_transition_count

def vertcial_transisions (img):
    vertical_transition_count = 0
    row_found_in = {}
    for c in range(img.shape[1]):
        for r in range(img.shape[0]-1):
            if img[r,c] != img[r+1,c] and not( r in row_found_in.keys()):
                row_found_in[r] = 1
                vertical_transition_count+=1
    return vertical_transition_count

def number_of_transitions(img):
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ht = horizontal_transitions(img)    
    vt = vertcial_transisions(img)
    return ht,vt

#Mufeed
def Center_of_mass(letters):
    properties = regionprops(letters, letters)
    center_of_mass = properties[0].centroid
    weighted_center_of_mass = properties[0].weighted_centroid
    return (center_of_mass[1]/letters.shape[0]), (center_of_mass[0]/letters.shape[1]) 
    
def Black_ink_histogram(letter):# check needed
    VP = getVerticalProjection(letter)
    VP_ink = [VP[i*len(VP)//20] for i in range(20)]
    HP = getHorizontalProjection(letter)
    HP_ink = [HP[i*len(HP)//20] for i in range(20)]
    return VP_ink, HP_ink

def Connected_Component(letter):
    ret, labels, stats,center = cv2.connectedComponentsWithStats(letter)
    return ret

#####################
#Salah
def ratio_of_colors(region1, region2, color1, color2):
    '''
    Gets ratios between 2 regions
    color1 = 0, 1
    color2 = 0, 1
    0 --> black
    1 --> white
    '''
    
    region1 = np.array(region1)
    region2 = np.array(region2)

    region1 = region1.flatten()
    region2 = region2.flatten()

    required1 = None
    required2 = None
    if color1 == 0:
        required1 = region1[region1 == 0]
        required1 = required1.shape[0]
    else:
        required1 = region1[region1 == 255]
        required1 = required1.shape[0]
    if color2 == 0:
        required2 = region2[region2 == 0]
        required2 = required2.shape[0]
    else:
        required2 = region2[region2 == 255]
        required2 = required2.shape[0]

    if required2 == 0:
        return float('nan')

    return required1/required2

def height_over_width(image):
    image = np.array(image)
    return image.shape[0]/image.shape[1]

def divide_image_to_four_regions(image):
    image = np.array(image)
    img_cols = image.shape[1]
    img_rows = image.shape[0]
    
    mid_col = img_cols//2
    mid_row = img_rows//2

    first_region  = image[0:mid_row,0:mid_col]
    second_region = image[0:mid_row,mid_col:]

    third_region  = image[mid_row+1:,0:mid_col+1]
    fourth_region = image[mid_row+1:,mid_col+1:]

    return first_region, second_region, third_region, fourth_region
#####################
def ratiosBlackWhite(letter):
    r1,r2,r3,r4 = divide_image_to_four_regions(letter)
    fig, ax = plt.subplots()
    f1  = ratio_of_colors(r1,r1,0,1)
    f2  = ratio_of_colors(r2,r2,0,1)
    f3  = ratio_of_colors(r3,r3,0,1)
    f4  = ratio_of_colors(r4,r4,0,1)

    f5  = ratio_of_colors(r1,r2,0,0)
    f6  = ratio_of_colors(r3,r4,0,0)
    f7  = ratio_of_colors(r1,r3,0,0)
    f8  = ratio_of_colors(r2,r4,0,0)
    f9  = ratio_of_colors(r1,r4,0,0)
    f10 = ratio_of_colors(r2,r3,0,0)

    return f1,f2,f3,f4,f5,f6,f7,f8,f9,f10

def count_holes(img, num_connected_parts):  # count number of holes in each character
    contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return 0
    elif abs(len(contours) - (num_connected_parts-1)) < 0:
        return 0
    return abs(len(contours) - (num_connected_parts-1))
