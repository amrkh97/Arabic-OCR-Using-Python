from Levenshtein import *

txt1 = open("output/text/coct1275.txt").read()
txt2 = open("predictions.txt").read()

print("distance:", distance(txt1,txt2)