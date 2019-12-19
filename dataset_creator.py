import os
import cv2
import numpy as np


def createArabicDictionary():
    D = {}
    D['ا'] = 0
    D['ب'] = 1
    D['ت'] = 2
    D['ث'] = 3
    D['ج'] = 4
    D['ح'] = 5
    D['خ'] = 6
    D['د'] = 7
    D['ذ'] = 8
    D['ر'] = 9
    D['ز'] = 10
    D['س'] = 11
    D['ش'] = 12
    D['ص'] = 13
    D['ض'] = 14
    D['ط'] = 15
    D['ظ'] = 16
    D['ع'] = 17
    D['غ'] = 18
    D['ف'] = 19
    D['ق'] = 20
    D['ك'] = 21
    D['ل'] = 22
    D['م'] = 23
    D['ن'] = 24
    D['ه'] = 25
    D['و'] = 26
    D['ي'] = 27
    D['لا'] = 29
    return D

def saveLettersToImages(letter,label):
    cv2.imwrite('../Dataset/{}.jpg'.format(label), letter)
    
def checkNumberOfSeparations():
    pass

def createDataSet(ArabicDictionary,images,labels):
    i = 0
    for image in images:
        label = ArabicDictionary[labels[i]]
        
        saveLettersToImages(image,str(label))
        i += 1

D = createArabicDictionary()
img = cv2.imread("./Test Data Set/image.png")
createDataSet(D, [img],['ي'])