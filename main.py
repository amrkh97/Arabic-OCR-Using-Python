import cv2
import time
import numpy as np
from scipy import stats
import feature_extractor as FE
from commonfunctions import *

start_time = time.time()

###################################################################

input_image = cv2.imread("./Test Data Set/image.png")

len_words = FE.preprocessIntoWords(input_image)

for line in len_words:
    
    BLI = FE.BaselineDetection(line)
    Detected_Words = np.flip(FE.DetectWords(line))
    
    for word in Detected_Words:  
        SegmentedWord = FE.extractFromWord(word,BLI)
        for k in SegmentedWord:
            show_images([k])
    
print("Running Time In Seconds: {0:.3f}".format(time.time() - start_time))