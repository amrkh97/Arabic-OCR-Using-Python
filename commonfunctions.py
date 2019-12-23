import cv2
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

def threshold(img):
    thresholded = np.copy(img)
    _, thresholded = cv2.threshold(thresholded, 127, 255, cv2.THRESH_BINARY_INV)
    thresholded = thresholded//255
    return thresholded