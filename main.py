from commonfunctions import *
import cv2
import feature_extractor as FE
import numpy as np
import time

start_time = time.time()

###################################################################

input_image = cv2.imread("./Pattern Data Set/scanned/csep1638.png")
if len(input_image.shape) == 3:
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

input_image = FE.correct_skew(input_image)
len_words = FE.DetectLines(input_image)

i=0
line = len_words[0]
for line in len_words:
    showImage(line)
    BLI = FE.BaselineDetection(line)
    MTI = FE.FindingMaximumTransitions(line, BLI)
    
    '''
    lineBGR = FE.returnToBGR(line)
    cv2.imshow("Detected BLI And MTI",lineBGR)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    
    print("Line: {} --> BLI: {}, MTI: {}".format(i,BLI,MTI))
    i += 1
    Detected_Words = FE.DetectWords(line)
    word = Detected_Words[0]
    ret, word = cv2.threshold(word, 127, 255, cv2.THRESH_BINARY_INV)
    word = word//255
    # TODO: Ask Amr about the above 2 lines
    CPI = FE.CutPointIdentification(line, word, MTI) # Algorithm 6
    for i in range (len(CPI)):
        word[MTI,int(CPI[i].CutIndex)] = 150
    show_images([word])
    i+=1

print("Running Time In Seconds: {0:.3f}".format(time.time() - start_time))