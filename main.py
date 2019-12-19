import cv2
import feature_extractor as FE
import matplotlib.pyplot as plt
import numpy as np

def showImage(img):
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_images(images,titles=None):
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image,title in zip(images,titles):
        a = fig.add_subplot(1,n_ims,n)
        if image.ndim == 2: 
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show()

input_image = cv2.imread("./Pattern Data Set/scanned/csep1638.png")
if len(input_image.shape) == 3:
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

input_image = FE.correct_skew(input_image)
len_words = FE.DetectLines(input_image)

i=0
line = len_words[0]
for line in len_words:
    showImage(line)
    BLI = FE.BaselineDetection(line)
    MTI = FE.FindingMaximumTransitions(line, BLI)
    print("Line: {} --> BLI: {}, MTI: {}".format(i,BLI,MTI))
    i += 1
    Detected_Words = FE.DetectWords(line)
    word = Detected_Words[0]
    ret, word = cv2.threshold(word, 127, 255, cv2.THRESH_BINARY_INV)
    word = word//255
    # TODO: Ask Amr about the above 2 lines
    CPI = FE.CutPointIdentification(line, word, MTI) # Algorithm 6
    for i in range (len(CPI)):
        word[MTI,int(CPI[i].CutIndex)] = 150
    show_images([word])