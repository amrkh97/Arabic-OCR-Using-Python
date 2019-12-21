import os
import cv2
import csv
import numpy as np
import pandas as pd
from io import StringIO



def get_words_from_file(path,fileName):
    wordList = []
    text = path+fileName
    t = np.loadtxt(text, dtype = 'str',encoding='utf-8', delimiter='\n')
    if t.shape == ():
        t = t.reshape(1,)
        for line in t[0].split(" "):
            wordList.append(line)
        return wordList
    for line in t:
        l = line.split(" ")
        wordList.extend(l)
    return wordList

def get_letters_from_word(wordList):
    lettersList = []
    for word in wordList:
        char = list(word)
        char.reverse()
        lettersList.append(char)
    return lettersList

def read_text_file(path,fileName):
    lis = get_words_from_file(path,fileName)
    lis2 = get_letters_from_word(lis)
    return lis2

#WIP
# Needs to be made with pandas and in the form of an iterator
def read_features_from_file(path, fileName):
    VP_HP_list = []
    labels_list = []
    with open('image_label_pair.csv') as file:
        rows = csv.reader(file, delimiter=',',quoting=csv.QUOTE_NONNUMERIC)
        for row in rows:
            label = row.pop(-1)
            labels_list.append(int(label))
            row = list(np.int_(row))
            VP_HP_list.append(row)
    return VP_HP_list, labels_list

#WIP
def processChunkToTrain(chunk):
    chunk  = np.array(chunk)
    for row in  chunk:
        print(row)
    


def pandasCSVHandler(fileName,chunkSize):
    
    for chunk in pd.read_csv(fileName,chunksize=chunkSize):
        processChunkToTrain(chunk)


#pandasCSVHandler('image_label_pair.csv',100000)