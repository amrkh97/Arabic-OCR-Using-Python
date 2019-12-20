import cv2
import time
import glob
import numpy as np
from scipy import stats
import feature_extractor as FE
from commonfunctions import *

start_time = time.time()

###################################################################

print("Running Time In Seconds: {0:.3f}".format(time.time() - start_time))

#WIP
def ReadAndSegment():
    Images_Path = './Test Data Set/'

    Number_Of_Files = 1           
    gen =  glob.iglob(Images_Path + "*.png")
    for i in range(Number_Of_Files):
        py = next(gen)
        input_image = cv2.imread(py)
        all_words = FE.extractSeparateLettersWholeImage(input_image)