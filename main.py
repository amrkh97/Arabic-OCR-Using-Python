import cv2
import time
import numpy as np
import feature_extractor as FE
from commonfunctions import *

start_time = time.time()

###################################################################

input_image = cv2.imread("./Pattern Data Set/scanned/csep1638.png")
if len(input_image.shape) == 3:
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

input_image = FE.correct_skew(input_image)
len_words = FE.DetectLines(input_image)

i=0
for line in len_words:
    
    #showImage(line)
    BLI = FE.BaselineDetection(line)
    MTI = FE.FindingMaximumTransitions(line, BLI)
    lineBGR = FE.returnToBGR(line)
    #cv2.line(lineBGR, (0,BLI), (lineBGR.shape[1],BLI), (255,0,0), 1) 
    #cv2.line(lineBGR, (0,MTI), (lineBGR.shape[1],MTI), (0,255,0), 1) 
    #showImage(lineBGR)
    print("Line: {} --> BLI: {}, MTI: {}".format(i,BLI,MTI))
    i += 1
    Detected_Words = np.flip(FE.DetectWords(line))
    
    word = Detected_Words[0]
    _, word = cv2.threshold(word, 127, 255, cv2.THRESH_BINARY_INV)
    
    CPI = FE.CutPointIdentification(line, word, MTI) # Algorithm 6
    
    for j in range(len(CPI)):
        word[MTI,int(CPI[j].CutIndex)] = 120
        
    show_images([word])
    #showImage(word)

print("Running Time In Seconds: {0:.3f}".format(time.time() - start_time))