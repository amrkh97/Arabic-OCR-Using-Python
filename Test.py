import csv
import cv2
import feature_extractor as FE
import glob
import numpy as npos
import os
import pandas as pd
import time

from commonfunctions import *
from scipy import stats


def ShowImageCV2(image):
    for img in image:
        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def save_letters_to_csv(letter):
    resized = cv2.resize(letter, (28, 28), interpolation=cv2.INTER_AREA)
    VP = FE.getVerticalProjection(resized)
    HP = FE.getHorizontalProjection(resized)
    concat = np.concatenate((VP, HP), axis=0)
    concat = concat.tolist()
    with open("image_label_pair_TEST.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(concat)

def pandasCSVHandler(fileName,chunkSize):
    print("Started Chuncking!")
    for chunk in pd.read_csv(fileName,chunksize=chunkSize):
        print(chunk)

def write_prediction_to_txt(words):
    '''
    words 2D Array
    '''
    if os.path.exists("predictions.txt"):
        os.remove("predictions.txt")
    file = open("predictions.txt","w",encoding='utf-8')
    
    for word in words:
        for letter in word:
            file.write(letter)
        file.write(" ")
    file.close()
    
    # Open file to user
    os.system("predictions.txt")

###################################################################

def test(path, number_of_files):
    gen = glob.iglob(path + "*.png")
    for i in range(number_of_files):
        py = next(gen)
        input_image = cv2.imread(py)
        all_words = FE.extractSeparateLettersWholeImage(input_image)
        for word in all_words:
            for letter in word:
                save_letters_to_csv(letter)

def main():
    start_time = time.time()

    path = './Pattern Data Set/scanned/'
    number_of_files = 1

    test(path, number_of_files)
    
    pandasCSVHandler("image_label_pair_TEST.csv", 1000)

    # write_prediction_to_txt(words)

    print("Running Time In Seconds: {0:.3f}".format(time.time() - start_time))


# main()
write_prediction_to_txt([["أ","ح","م","د",], ["ع","م","ر","و"], ["ح","س","ي","ن"]])