import cv2
import csv
import time
import glob
import numpy as np
import read_files as RF
import feature_extractor as FE
import dataset_creator as DC
 
from scipy import stats
from commonfunctions import *



start_time = time.time()

###################################################################

#WIP
def ReadAndSegment(ii):
    Path = './Test Data Set/'

    Number_Of_Files = 1
    #Number_Of_Files = 1           
    gen =  glob.iglob(Path+ "*.png")
    for i in range(Number_Of_Files):
        py = next(gen)
        #print("Currently Segmenting: ",py)
        input_image = cv2.imread(py)
        splitted = py.split("\\")
        splitted = splitted[1].split(".")
        splitted = splitted[0]
        splitted += ".txt"
        #input_image = cv2.imread('./Pattern Data Set/scanned/capr102.png')
        list_of_letters = RF.read_text_file(Path,splitted)
        all_words = FE.extractSeparateLettersWholeImage(input_image)
        DC.createDataSet(all_words,list_of_letters)

        ii += 1
    print(ii)

j = 0        
ReadAndSegment(j)
# reading from file for features
# Path = './Test Data Set/'
# Name = 'image_label_pair.csv'
# VP_HP_list, labels_list = RF.read_features_from_file(Path,Name)


print("Running Time In Seconds: {0:.3f}".format(time.time() - start_time))

