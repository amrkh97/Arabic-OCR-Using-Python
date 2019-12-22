import cv2
import feature_extractor as fe

img = cv2.imread('Dataset/20.png')

hT , vT = fe.number_of_transitions(img)
print(hT,vT)

print("***************************************")