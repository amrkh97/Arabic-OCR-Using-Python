import cv2
import numpy as np
from scipy import stats


def getVerticalProjection(img):
    return (np.sum(img, axis = 0))//255

def getHorizontalProjection(img):
    return (np.sum(img, axis = 1))//255

def DetectWords(line_img):
    pass
