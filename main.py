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


for line in len_words:
    
    BLI = FE.BaselineDetection(line)
    Detected_Words = np.flip(FE.DetectWords(line))
    
    for word in Detected_Words:        
        
        word = threshold(word)
        word = word * 255
        
        wordCopy = FE.amrsFunction(word,BLI)
        listOfSegmentations = FE.extractLettersFromWord(wordCopy)
        SegmentedWord = FE.showImagesFromSegments(word,listOfSegmentations)
        for k in SegmentedWord:
            show_images([k])
    
print("Running Time In Seconds: {0:.3f}".format(time.time() - start_time))