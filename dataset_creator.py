import os
import cv2
import numpy as np
import feature_extractor as FE
from commonfunctions import *
from read_files import read_text_file
import csv

def ShowImageCV2(image):
    for img in image:
        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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
    D['لا'] = 28
    return D

def returnToArabicDictionary():
    D = {}
    D[0] = 'ا'
    D[1] = 'ب'
    D[2] = 'ت'
    D[3] = 'ث'
    D[4] = 'ج'
    D[5] = 'ح'
    D[6] = 'خ'
    D[7] = 'د'
    D[8] = 'ذ'
    D[9] = 'ر'
    D[10] = 'ز'
    D[11] = 'س'
    D[12] = 'ش'
    D[13] = 'ص'
    D[14] = 'ض'
    D[15] = 'ط'
    D[16] = 'ظ'
    D[17] = 'ع'
    D[18] = 'غ'
    D[19] = 'ف'
    D[20] = 'ق'
    D[21] = 'ك'
    D[22] = 'ل'
    D[23] = 'م'
    D[24] = 'ن'
    D[25] = 'ه'
    D[26] = 'و'
    D[27] = 'ي'
    D[28] = 'لا'
    return D

def saveLettersToImages(letter,label):
    
    hw = FE.height_over_width(letter)
    
    letter = cv2.resize(letter, (28,28), interpolation = cv2.INTER_AREA)
    
    VP_ink,HP_ink = FE.Black_ink_histogram(letter)
    Com1,Com2 = FE.Center_of_mass(letter)
    CC = FE.Connected_Component(letter)
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
    
    concat.append(label)
    with open("image_label_pair.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(concat)
  
def checkNumberOfSeparations(wordSeparationList,lettersOfWordList): #Expecting an ndarray and a list
    '''
    wordSeaparationList = single word segmentation
    lettersOfWordList = letters of word from text file
    '''
    checkBool = False
    
    numberofSegments = len(wordSeparationList)
    lettersOfWordList = np.array(lettersOfWordList)
    actualNumber = len(lettersOfWordList)
    if numberofSegments == actualNumber: # No action needed as number of segments same as number of letters
        checkBool = True
        return wordSeparationList,checkBool 
    
    if numberofSegments < actualNumber: # This means that the BLI value caused an error and the word was under segmented
        return wordSeparationList,checkBool
    
    if numberofSegments > actualNumber: #Oversegmented word but may be handled
        return wordSeparationList,checkBool

def createDataSet(images,labels):
    D = createArabicDictionary()
    i = 0
    for word in images:
        j = 0
        segmented_list, no_segmentation_error = checkNumberOfSeparations(word, labels[i])
        
        if no_segmentation_error:
            for l in labels[i]:
                label = D[l]
                saveLettersToImages(word[j], label)
                j+=1
        i += 1