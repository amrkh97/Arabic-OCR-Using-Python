import cv2
import time
import numpy as np
from scipy import stats
import feature_extractor as FE
from commonfunctions import *

start_time = time.time()

###################################################################

input_image = cv2.imread("./Test Data Set/image.png")

all_words = FE.extractSeparateLettersWholeImage(input_image)

print("Running Time In Seconds: {0:.3f}".format(time.time() - start_time))
