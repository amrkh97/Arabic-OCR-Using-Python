import os
import cv2
import numpy as np



def read_text_file(fileName):
    wordsList = []
    with open('./Test Data Set/'+fileName+'.txt','r', encoding ='utf-8') as f:
        for line in f:
            for word in line.split():
                wordLetters = []
                for letter in word:
                    wordLetters.append(letter)
                #wordLetters.reverse()
                wordsList.append(wordLetters)
    return wordsList


filename ='test'
allLetters = read_text_file(filename)
print(allLetters)
