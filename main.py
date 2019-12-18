import cv2
import numpy as np
import feature_extractor as FE


input_image = cv2.imread("./Pattern Data Set/scanned/csep1638.png")
if len(input_image.shape) == 3:
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

input_image = FE.correct_skew(input_image)
len_words = FE.DetectLines(input_image)

i=0
for line in len_words:
    
    BLI = FE.BaselineDetection(line)
    MTI = FE.FindingMaximumTransitions(line, BLI)
    print("Line: {} --> BLI: {}, MTI: {}".format(i,BLI,MTI))
    i+=1
    #Detected_Words = FE.DetectWords(line) 
    