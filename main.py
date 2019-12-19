import cv2
import time
import numpy as np
from scipy import stats
import feature_extractor as FE
from commonfunctions import *

start_time = time.time()

###################################################################

input_image = cv2.imread("./Test Data Set/image.png")
if len(input_image.shape) == 3:
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

input_image = FE.correct_skew(input_image)
len_words = FE.DetectLines(input_image)

i=0
for line in len_words:
    BLI = FE.BaselineDetection(line)
    MTI = FE.FindingMaximumTransitions(line, BLI)
    lineBGR = FE.returnToBGR(line)
    # cv2.line(lineBGR, (0,BLI), (lineBGR.shape[1],BLI), (255,0,0), 1) 
    # cv2.line(lineBGR, (0,MTI), (lineBGR.shape[1],MTI), (0,255,0), 1) 
    # showImage(lineBGR)
    print("Line: {} --> BLI: {}, MTI: {}".format(i,BLI,MTI))
    i += 1
    Detected_Words = np.flip(FE.DetectWords(line))
    # W = FE.removeDots(word)
    line = threshold(line)
    for word in Detected_Words:        
        word = threshold(word)
        SRL = FE.CutPointIdentification(line, word, MTI) # Algorithm 6
        word = word * 255
        for i in range (len(SRL)):
            word[:,int(SRL[i].CutIndex)] = 150
        show_images([word])
    
    line = line * 255
<<<<<<< HEAD
    # word = word * 255
    '''
    for i in range (len(CPI)):
        word[:,int(CPI[i].CutIndex)] = 150
    show_images([word])
    '''
    MFV = 0
    MFV = stats.mode(word.tolist())[0][0]
    print(len(CPI))
    TrueCuts = FE.algo7(line,word,CPI,BLI,MTI,MFV)
    print(len(TrueCuts))
=======
    word = word * 255
    MFV = stats.mode(word.tolist())[0][0]
    
    TrueCuts = FE.algo7(line,word,SRL,BLI,MTI,MFV)
>>>>>>> 5858492d2aa82926a5d8a7a7021bd55b613e712b
    for i in range (len(TrueCuts)):
        word[:,TrueCuts[i].CutIndex] = 0
    show_images([word])
    
    
print("Running Time In Seconds: {0:.3f}".format(time.time() - start_time))