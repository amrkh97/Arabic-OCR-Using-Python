import cv2
import csv
import time
import glob
import torch
import numpy as np
import read_files as RF
import neural_network as NN
import feature_extractor as FE
import dataset_creator as DC
from scipy import stats
from commonfunctions import *

#################################################################################

start_time = time.time()

chunkSize = 1000 #Read 1000 lines at a time to handle memory restrictions and errors
inputSize =  55  #Size of feature file

div = torch.device('cuda')
model = NN.createNN(inputSize)
model = model.to(device=div)
RF.pandasCSVHandler(model,'image_label_pair.csv',chunkSize)

print("Running Time In Seconds: {0:.3f}".format(time.time() - start_time))

