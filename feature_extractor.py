import cv2
import imutils
import numpy as np
import word2subwords as w2s
import matplotlib.pyplot as plt
from scipy import stats
from imutils import contours
from scipy import ndimage


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

def removeDots(img):
    copy_img = img.copy()
    subword_cnts = w2s.string2subwords(copy_img, delete_diacritics=True)
    return w2s.draw_subwords(copy_img.shape, subword_cnts)

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

# added func for algo 7
def pathBetweenStartandEnd(vp):
    count = np.count_nonzero(vp)
    if count < len(vp):
        return True
    return False

def Count_connected_parts(img):  # this function returns the number of connected parts given the binary image of any letter
    labeled, nr_objects = ndimage.label(img > 0)  # 100 is the threshold but in case of binary image given (0,1) it will change
    return nr_objects


def count_holes(img, num_connected_parts):  # count number of holes in each character
    contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return abs(len(contours) - num_connected_parts)

def DetectBaselineBetweenStartAndEnd(Word, BLI, Start, End):#End is left
    if np.sum( Word[BLI,End:Start] ) == 0:
        return True #no path found
    return False

def DistanceBetweenTwoPoints(x2,x1):
    dist = np.abs(x2 - x1) 
    return dist

def CheckLine19Alg7(SRL,SR, NextCutIndex, VP, Word,MTI,BLI):
    LeftPixelCol = SR.EndIndex
    TopPixelIndex = 0
    for i in range(MTI,MTI-20,-1):
        if Word[i-1,LeftPixelCol] == 0:
            TopPixelIndex = i
            break
    Dist1 = DistanceBetweenTwoPoints( TopPixelIndex,BLI )
    Dist2 = DistanceBetweenTwoPoints( MTI,BLI )
    if ( SR==SRL[0] and VP[NextCutIndex] == 0) or ( Dist1 < (0.5*Dist2) ) :
        return True
    return False

def CheckStroke(Word, SR,NextCut, CurrentCut, PreviousCut, MTI,BLI):
    HPAbove = getHorizontalProjection( Word[0:BLI,:] )
    HPBelow = getHorizontalProjection( Word[BLI:,:] )  
    SHPB = np.sum(HPBelow)
    SHPA = np.sum(HPAbove)
    LeftPixelCol = SR.EndIndex
    TopPixelIndex = 0
    for i in range(MTI,MTI-20,-1):
        if Word[i-1,LeftPixelCol] == 0:
            TopPixelIndex = i
            break
            
    Dist1 = DistanceBetweenTwoPoints( TopPixelIndex,BLI )
    HP = getHorizontalProjection(Word)
    HPList = HP.tolist()
    HPMode = max(set(HPList), key = HPList.count) 
    HPList.sort()
    SecondPeakValue = HPList[-2]
    
    VP=getVerticalProjection(Word)
    VPList = VP.tolist()
    MFV = max(set(VPList), key = VPList.count) 
    
    Holes = DetectHoles(Word, NextCut, CurrentCut, PreviousCut, MTI)
    if (SHPA > SHPB) and (Dist1 < (2*SecondPeakValue) ) and (HPMode == MFV) and (~Holes) :
        return True
    return False

def DetectHoles(Word, NextCut, CurrentCut, PreviousCut, MTI):#next is left, previous is right
    LefPixelIndex = 0
    for i in range(NextCut, PreviousCut, 1):
        if Word[MTI, i] == 1:
            LefPixelIndex = i
            break
    
    RightPixelIndex = 0
    for i in range(PreviousCut, NextCut, -1):
        if Word[MTI, i] == 1:
            RightPixelIndex = i
            break
    
    UpPixelIndex = 0
    for i in range(MTI, MTI - 10, -1):
        if Word[i, CurrentCut] == 1:
            UpPixelIndex = i
            break
    
    DownPixelIndex = 0
    for i in range(MTI, MTI + 10, 1):
        if Word[i, CurrentCut] == 1:
            DownPixelIndex = i
            break
    
    if ( np.abs(LefPixelIndex - RightPixelIndex) <=5 ) and ( np.abs(UpPixelIndex - DownPixelIndex ) <=5 ):
        return True
    else:
        return False
    
def DetectDots(word, start_index, end_index):
    HP = getHorizontalProjection[word[start_index:end_index, :]]
    for H in HP:
        H[H>0] = 1
        diffArray = np.diff(H)
        absolute = np.absolute(diffArray)
        sume = np.sum(absolute)
        if sume%4 == 0:
            return True
    return False

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
    #plt.plot(HP)
    #plt.show()
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
    VP = getVerticalProjection(LineImage)
    MFV = stats.mode(VP.tolist())[0][0]
    SeaparationRegions = []
    BeginIndex = 0
    EndIndex = len(VP)
    # for i in VP:
    #     if i == 0:
    #         BeginIndex+=1

    # for i in range(-1,-30,-1):
    #     if VP[i] == 0:
    #         EndIndex -= 1

    # VP = VP[BeginIndex:EndIndex]

    while i <= word.shape[1]:
        # Line 8
        if word[MTI, i] == 1 and FLAG == 0:
            SR = SeaparationRegion()
            SR.setEndIndex(i)
            FLAG = 1
        if i == (word.shape[1] - 1):
            break
        # Line 12
        elif word[MTI, i] != 1 and FLAG == 1:
            SR.setStartIndex(i)
            MidIndex = int((SR.EndIndex + SR.StartIndex)/2)
            VP = np.array(VP)
            # k is an array of indices
            k_equal_zero = np.where(VP[SR.StartIndex:SR.EndIndex] == 0)
            k_equal_zero = k_equal_zero[0]
            
            k_less_than_MFV_EndIndex = np.where(VP <= MFV) and np.where(VP < SR.EndIndex)
            k_less_than_MFV_EndIndex = k_less_than_MFV_EndIndex[0]

            k_less_than_MFV = np.where(VP[SR.EndIndex:SR.StartIndex] <= MFV)
            k_less_than_MFV = k_less_than_MFV[0]
            
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
#WIP
def algo7(line,Word,srl,baselineIndex,maxtransisitionIndex,mfv):
    i = 0
    word = np.copy(Word)
    validsaperationRegion = []
    VP = getVerticalProjection(word)
    while i < len(srl):
        sr = srl[i]
        sr.CutIndex -= sr.EndIndex 
        if pathBetweenStartandEnd(VP[sr.EndIndex:sr.StartIndex]):
            sr.CutIndex += sr.EndIndex
            validsaperationRegion.append(sr)
            i += 1
        elif  i !=0:
            if (i+1) < len(srl):
                Previous_sr = srl[i-1]
                Next_sr = srl[i+1]
                labels = Count_connected_parts(word[:, Previous_sr.CutIndex:Next_sr.CutIndex])
                numberHoles = count_holes(word[:, Previous_sr.CutIndex:Next_sr.CutIndex],labels)
                if numberHoles != 0:
                    i += 1
        elif not(DetectBaselineBetweenStartAndEnd(word,baselineIndex, sr.EndIndex, sr.StartIndex)):
            SHPB = len(getHorizontalProjection(word[0: baselineIndex,:]))
            SHPA = len(getHorizontalProjection(word[baselineIndex:,:]))
            if SHPB > SHPA:
                i+=1
            elif VP[sr.CutIndex] < mfv:
                sr.CutIndex += sr.EndIndex
                validsaperationRegion.append(sr)
                i+=1
            else: 
                i+=1
        elif (i+1) < len(srl):
            if i == len(srl)-1 or (srl[i+1].CutIndex == 0 and CheckLine19Alg7(srl,sr,srl[i+1].CutIndex,VP,word,maxtransisitionIndex,baselineIndex)):
                i+=1
        elif (i+1) < len(srl) and (i-1) > 0:
            if CheckStroke(word,sr,srl[i+1].CutIndex,sr.CutIndex,srl[i-1].CutIndex,maxtransisitionIndex,baselineIndex) != True:
                if DetectBaselineBetweenStartAndEnd(word,baselineIndex, srl[i+1].EndIndex, srl[i+1].StartIndex) and srl[i+1].CutIndex <= mfv:
                    i+=1
                else:
                    sr.CutIndex += sr.EndIndex
                    validsaperationRegion.append(sr)
                    i+=1
        elif (i+1) < len(srl) and (i-1) > 0:
            if CheckStroke(word,sr,srl[i+1].CutIndex ,sr.CutIndex ,srl[i-1].CutIndex,maxtransisitionIndex,baselineIndex):
                sr.CutIndex += sr.EndIndex
                validsaperationRegion.append(sr)
                i+=1
        elif (i+1) < len(srl) and (i-1) > 0:
            if CheckStroke(word,sr,srl[i+1].CutIndex ,sr.CutIndex ,srl[i-1].CutIndex,maxtransisitionIndex,baselineIndex) and DetectDots(word,sr.CutIndex, srl[i+1].CutIndex):
                sr.CutIndex += sr.EndIndex
                validsaperationRegion.append(sr)
                i+=1
        elif (i+1) < len(srl) and (i+2) < len(srl) and (i+3) < len(srl):
            if CheckStroke(word,sr,srl[i+1].CutIndex ,sr.CutIndex ,srl[i-1].CutIndex,maxtransisitionIndex,baselineIndex) and (DetectDots(word,sr.CutIndex, srl[i+1].CutIndex) == False):
                if CheckStroke(word,sr,srl[i+2].CutIndex, sr.CutIndex[i+1], srl[i].CutIndex,maxtransisitionIndex,baselineIndex) and (DetectDots(word,srl[i+1].CutIndex, srl[i+2].CutIndex) == False):
                    sr.CutIndex += sr.EndIndex
                    validsaperationRegion.append(sr)
                    i+=3
                if (CheckStroke(word,sr,srl[i+2].CutIndex, sr.CutIndex[i+1], srl[i].CutIndex,maxtransisitionIndex,baselineIndex) and (DetectDots(word,srl[i+1].CutIndex, srl[i+2].CutIndex))) and (CheckStroke(word,sr,srl[i+3].CutIndex, sr.CutIndex[i+2], srl[i+1].CutIndex,maxtransisitionIndex,baselineIndex) and (DetectDots(word,srl[i+2].CutIndex, srl[i+3].CutIndex)) == False):
                    sr.CutIndex += sr.EndIndex
                    validsaperationRegion.append(sr)
                    i+=3
                if CheckStroke(word,sr,srl[i+2].CutIndex, sr.CutIndex[i+1], srl[i].CutIndex,maxtransisitionIndex,baselineIndex) and ((DetectDots(word,srl[i+1].CutIndex, srl[i+2].CutIndex) == False) or (DetectDots(word,srl[i+1].CutIndex, srl[i+2].CutIndex))):
                    i += 1
        else: 
            i+=1
        # elif VP[sr.CutIndex] == 0: 
        #     sr.CutIndex += sr.EndIndex 
        #     validsaperationRegion.append(sr)
        #     i += 1
    return validsaperationRegion


####################################################################

def amrsFunction(word,BLI):
    VP = getVerticalProjection(word)
    #plt.plot(VP)
    #plt.show()
    copyWord = np.copy(word)
    VP[VP < 2] = 0
    #copyWord[:BLI-1, VP == 0] = 0
    copyWord[BLI, VP == 0] = 0
    return copyWord

def amrsExtractLettersFromWords():
    pass