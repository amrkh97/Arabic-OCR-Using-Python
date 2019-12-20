import cv2
import time
import glob
import numpy as np
import read_files as RF
import feature_extractor as FE
 
from scipy import stats
from commonfunctions import *
from read_files import read_text_file


start_time = time.time()

###################################################################

#WIP
def ReadAndSegment(ii):
    Path = './Pattern Data Set/scanned/'

    Number_Of_Files = 1000
    #Number_Of_Files = 1           
    gen =  glob.iglob(Path+ "*.png")
    for i in range(Number_Of_Files):
        py = next(gen)
        #print("Currently Segmenting: ",py)
        input_image = cv2.imread(py)
        #input_image = cv2.imread('./Pattern Data Set/scanned/capr102.png')
        #RF.read_text_file(Path+'/text/',)
        all_words = FE.extractSeparateLettersWholeImage(input_image)
        ii += 1
    print(ii)

j = 0        
ReadAndSegment(j)


print("Running Time In Seconds: {0:.3f}".format(time.time() - start_time))

