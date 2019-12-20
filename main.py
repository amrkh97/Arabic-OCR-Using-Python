import cv2
import time
import glob
import numpy as np
import read_files as RF
import feature_extractor as FE
 
from scipy import stats
from commonfunctions import *

start_time = time.time()

###################################################################

#WIP
def ReadAndSegment():
    Path = './Pattern Data Set/scanned/'

    Number_Of_Files = 1           
    gen =  glob.iglob(Path+ "*.png")
    for i in range(Number_Of_Files):
        py = next(gen)
        #input_image = cv2.imread(py)
        input_image = cv2.imread('./Pattern Data Set/scanned/csep1638.png')
        #RF.read_text_file(Path+'/text/',)
        all_words = FE.extractSeparateLettersWholeImage(input_image)
        
ReadAndSegment()


print("Running Time In Seconds: {0:.3f}".format(time.time() - start_time))

