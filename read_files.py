import os
import cv2
import csv
import numpy as np
import pandas as pd
import neural_network as NN
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

def processChunkToTrain(chunk,model):
    chunk  = np.array(chunk)
    featuresList = chunk[:,:55]
    labelsList = chunk[:,56]
    
    finalList  = zip(featuresList, labelsList)
    finalList2 = zip(featuresList, labelsList)
    finalList3 = zip(featuresList, labelsList)
    finalList4 = zip(featuresList, labelsList)
    finalList5 = zip(featuresList, labelsList)
    
    NN.TrainNN(model,finalList,finalList2,finalList3,finalList4,finalList5)
    print("Ended Training the chunk!")    


def pandasCSVHandler(model,fileName,chunkSize):
    print("Started Chuncking!")
    for chunk in pd.read_csv(fileName,chunksize=chunkSize):
        processChunkToTrain(chunk,model)
        
        #model = NN.load_checkpoint('trained_model.pth')
