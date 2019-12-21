import cv2
import csv
import time
import glob
import numpy as np
import read_files as RF
import neural_network as NN
import feature_extractor as FE
import dataset_creator as DC
from scipy import stats
from commonfunctions import *

#################################################################################

start_time = time.time()

chunkSize = 10000 #Read 5000 lines at a time to handle memory restrictions and errors
inputSize =  55  #Size of feature file


model = NN.createNN(inputSize)
RF.pandasCSVHandler(model,'image_label_pair.csv',chunkSize)

print("Running Time In Seconds: {0:.3f}".format(time.time() - start_time))
