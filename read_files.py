import os
import cv2
import numpy as np
from io import StringIO



def read_text_file(path,fileName):
    LettersList = []
    text = path+fileName+'.txt'
    t = np.loadtxt(text, dtype = 'str',encoding='utf-8', delimiter=' ')
    for word in t:
        LettersList.append(list(word))
        # l = list(word)
        # l.reverse()
        # LettersList.append(l)
    return LettersList

path = './Test Data Set/'
fileName = 'test'
print(read_text_file(path,fileName))