import sys
import re
import numpy as np
import cv2
from diacritics_classification import classify_diacritics, bound_diacritics, compute_mc_coords


def extract_subword_imgs (img_shape, subword_cnts):
    res = []
    for cnts in subword_cnts:
        img = np.ones(img_shape, dtype=np.uint8) * 255
        cv2.drawContours(img, cnts, -1, 0, -1)
        res.append(img)
    return res

def draw_subwords (img_shape, subword_cnts):
    img = np.zeros(img_shape, dtype=np.uint8) * 255
    color = (255, 255, 255)
    prev_color = -1
    for cnts in subword_cnts:
        cv2.drawContours(img, cnts, -1, color, -1)
    return img


def natural_sort_key (x):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    res = [convert(c) for c in re.split('([0-9]+)', x)]
    return res



def find_contours (img_bw):
    cnts, _ = cv2.findContours((255 - img_bw).copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if c.shape[0] > 2]
    return cnts


def get_pxl_labels (img_shape, cnts):
    pxl_labels = np.ones(img_shape, dtype=np.int32) * -1
    for i in range(len(cnts)):
        cv2.drawContours(pxl_labels, cnts, i, i, -1)
    return pxl_labels


def get_subword_cnts (cnts, secondary2primary):
    # Get indices of primary components
    num_cnts = len(cnts)
    secondary_labels = secondary2primary.keys()
    primary_labels = set(range(num_cnts)) - set(secondary_labels)
    # Get list of contours for each subword
    label2cnts = {l: [cnts[l]] for l in primary_labels}
    for sl, pl in secondary2primary.items():
        label2cnts[pl].append(cnts[sl])
    subword_cnts = label2cnts.values()
    return subword_cnts


def sort_subwords (subword_cnts):
    subword_primary_cnts = [scnts[0] for scnts in subword_cnts]
    mc_coords = compute_mc_coords(subword_primary_cnts)
    
    num_subwords = len(subword_cnts)
    
    subword_cnts=list(subword_cnts)
    tmp = [(subword_cnts[i], mc_coords[i][0]) for i in range(num_subwords)]
    tmp.sort(key=lambda x: x[1], reverse=True)
    sorted_subword_cnts = [x[0] for x in tmp]
    return sorted_subword_cnts

def string2subwords (img, delete_diacritics=False):
    
    cnts = find_contours(img)
    # Make labeled img
    pxl_labels = get_pxl_labels(img.shape, cnts)
    # Classify to primary and secondary components
    thresh = 0.15
    is_primary = classify_diacritics(img, cnts, pxl_labels, thresh)

    if delete_diacritics:
        cnts = [cnt for i, cnt in enumerate(cnts) if is_primary[i]]
        secondary2primary = {}
    else:
        # Bound secondary components to primary
        secondary2primary = bound_diacritics(pxl_labels, cnts, is_primary)
    # Get list of contours for each subword
    # subwords are sorted through x coord
    subword_cnts = get_subword_cnts(cnts, secondary2primary)
    subword_cnts = sort_subwords(subword_cnts)
    
    return subword_cnts
