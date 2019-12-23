import csv
import cv2
import feature_extractor as FE
import glob
import numpy as np
import os
import pandas as pd
import time
import torch
import neural_network as NN
import NN2
from commonfunctions import *
from scipy import stats


def save_letters_to_csv(letter):
    hw = FE.height_over_width(letter)
    
    letter = cv2.resize(letter, (28,28), interpolation = cv2.INTER_AREA)
    
    VP_ink,HP_ink = FE.Black_ink_histogram(letter)
    Com1,Com2 = FE.Center_of_mass(letter)
    CC = FE.Connected_Component(letter)
    CH = FE.count_holes(letter,CC)
    r1,r2,r3,r4,r5,r6,r7,r8,r9,r10 = FE.ratiosBlackWhite(letter)
    HorizontalTransitions,VerticalTransitions = FE.number_of_transitions(letter)
    concat = []
    #concat = [*VP_ink, *HP_ink] #28+28 = 56
    concat.append(Com1) #1
    concat.append(Com2) #1
    concat.append(CC) #1
    concat.append(r1) #1
    concat.append(r2) #1
    concat.append(r3) #1
    concat.append(r4) #1
    concat.append(r5) #1
    concat.append(r6) #1
    concat.append(r7) #1
    concat.append(r8) #1
    concat.append(r9) #1
    concat.append(r10) #1
    concat.append(HorizontalTransitions) #1
    concat.append(VerticalTransitions) #1
    concat.append(hw) #1
    concat.append(CH) #1
    with open("image_label_pair_TEST.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(concat)

def pandasCSVHandler(fileName):
    
    FinalListForWriting = []
    chunk = pd.read_csv(fileName)
    chunk = chunk.values
    X = chunk.astype(float)
    listW = NN2.model_prediction(X)
    listW.reverse()
    listW = [listW,]
    FinalListForWriting.append(listW)
    
    return FinalListForWriting

def write_prediction_to_txt(words,i):
    '''
    words 1D Array
    '''
    # print(word)
    file = open("output/text/test_{}.txt".format(str(i+1)),"a",encoding='utf-8')
    
    for letter in words[0][0]:
        file.write(letter)
    file.close()

###################################################################

def test(path, number_of_files):
    FinalListForWriting = []
    if os.path.exists("image_label_pair_TEST.csv"):
        os.remove("image_label_pair_TEST.csv")
    with open("image_label_pair_TEST.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
    gen = glob.iglob(path + "*.png")
    for i in range(number_of_files):
        start_time = time.time()
        py = next(gen)
        input_image = cv2.imread(py)
        all_words = FE.extractSeparateLettersWholeImage(input_image)
        for word in all_words:
            if os.path.exists("image_label_pair_TEST.csv"):
                os.remove("image_label_pair_TEST.csv")
            with open("image_label_pair_TEST.csv", 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
            for letter in word:
                save_letters_to_csv(letter)
            # Single Word
                FinalListForWriting = pandasCSVHandler("image_label_pair_TEST.csv") 
            
            write_prediction_to_txt(FinalListForWriting,i)
            FinalListForWriting = []
            file = open("output/text/test_{}.txt".format(str(i+1)),"a",encoding='utf-8')
            file.write(' ')
            file.close()
        file = open("output/running_time.txt","a",encoding='utf-8')
        runTime = time.time() - start_time
        file.write(str(runTime))
        file.write('\n')
        file.close()

def main():
    start_time = time.time()

    path = './test/'
    number_of_files = 11
    #changes size
    #model = NN.createNN(17)
    #model.load_state_dict(torch.load('trained_model.pth',   map_location=torch.device('cpu')))
    #model.eval()
    
    test(path, number_of_files)
    runTime = time.time() - start_time
    print("Running Time In Seconds: {0:.3f}".format(runTime))
    file = open("output/runtime.txt","w",encoding='utf-8')
    file.write(str(runTime))
    file.close()

main()