import csv
import cv2
import dataset_creator as DC
import feature_extractor as FE
import glob
import numpy as np
import os
import read_files as RF
import time
import platform

from scipy import stats
from commonfunctions import *



start_time = time.time()

###################################################################

#WIP
def ReadAndSegment(ii):
    Path = './Dataset/'
    textPath = './Test Data Set/'
    
    Number_Of_Files = 12800
    #Number_Of_Files = 1           
    gen =  glob.iglob(Path+ "*.png")
    for i in range(Number_Of_Files):
        py = next(gen)
        input_image = cv2.imread(py)
        splitted = None
        if platform.system() == "Windows":
            splitted = py.split("\\")
            splitted = splitted[1].split(".")
        else:
            splitted = py.split("/")
            splitted = splitted[3].split(".")
        splitted = splitted[0]
        splitted += ".txt"
        list_of_letters = RF.read_text_file(textPath,splitted)
        all_words = FE.extractSeparateLettersWholeImage(input_image)
        DC.createDataSet(all_words,list_of_letters)

        ii += 1
        print("Ended Images Index: ",str(ii))

j = 0        
ReadAndSegment(j)



print("Running Time In Seconds: {0:.3f}".format(time.time() - start_time))

