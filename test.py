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
from commonfunctions import *
from scipy import stats


def ShowImageCV2(image):
    for img in image:
        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def save_letters_to_csv(letter):
    hw = FE.height_over_width(letter)
    
    letter = cv2.resize(letter, (28,28), interpolation = cv2.INTER_AREA)
    
    VP_ink,HP_ink = FE.Black_ink_histogram(letter)
    Com1,Com2 = FE.Center_of_mass(letter)
    CC = FE.Connected_Component(letter)
    CH = FE.count_holes(letter,CC)
    r1,r2,r3,r4,r5,r6,r7,r8,r9,r10 = FE.ratiosBlackWhite(letter)
    HorizontalTransitions,VerticalTransitions = FE.number_of_transitions(letter)
    
    concat = [*VP_ink, *HP_ink] #28+28 = 56
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

def pandasCSVHandler(fileName,chunkSize,model):
    print("Started Chuncking!")
    FinalListForWriting = []
    for chunk in pd.read_csv(fileName,chunksize=chunkSize):
        chunk = chunk.replace(np.nan, 0)
        c = np.array(chunk)
        FinalListForWriting.append(NN.TestNN(model,c[0,:]))
    
    return FinalListForWriting

def write_prediction_to_txt(word):
    '''
    words 1D Array
    '''
    file = open("predictions.txt","w",encoding='utf-8')
    
    # for word in words:
        #for letter in word:
            #file.write(letter)
        #file.write(" ")
        # file.write(word)
        #file.write(" ")
    file.write(word)
    file.write(" ")
    file.close()
    
    # Open file to user
    os.system("predictions.txt")

###################################################################

def test(path, number_of_files, model):
    if os.path.exists("image_label_pair_TEST.csv"):
        os.remove("image_label_pair_TEST.csv")
    gen = glob.iglob(path + "*.png")
    for i in range(number_of_files):
        py = next(gen)
        input_image = cv2.imread(py)
        all_words = FE.extractSeparateLettersWholeImage(input_image)
        for word in all_words:
            for letter in word:
                save_letters_to_csv(letter)
            # Single Word
            FinalListForWriting = pandasCSVHandler("image_label_pair_TEST.csv", 1, model)
            write_prediction_to_txt(FinalListForWriting)

def main():
    start_time = time.time()

    path = './Pattern Data Set/scanned/'
    number_of_files = 1
    model = NN.createNN(56)
    model.load_state_dict(torch.load('trained_model.pth'))
    model.eval()
     
    test(path, number_of_files)
    
    test(path, number_of_files, model)
    
    AllWordsConcated = pandasCSVHandler("test.csv", 1,model) #I won't know where the end of the word is
    print(AllWordsConcated)
    # write_prediction_to_txt(words)

    print("Running Time In Seconds: {0:.3f}".format(time.time() - start_time))


main()
# write_prediction_to_txt([["أ","ح","م","د",], ["ع","م","ر","و"], ["ح","س","ي","ن"]])