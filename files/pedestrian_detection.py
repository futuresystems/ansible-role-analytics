# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

mypath = "dataset/INRIAPerson/Test/pos/"
onlyfiles = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
args = {}
args['images'] = onlyfiles
# loop over the image paths

def run():
    cnt = 0
    for imagePath in paths.list_images(mypath):
        # load the image and resize it to (1) reduce detection time
        # and (2) improve detection accuracy
        image = cv2.imread(imagePath)
        image = imutils.resize(image, width=min(400, image.shape[1]))
        orig = image.copy()

        # detect people in the image
        (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
            padding=(8, 8), scale=1.05)

        # draw the original bounding boxes
        for (x, y, w, h) in rects:
            cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # apply non-maxima suppression to the bounding boxes using a
        # fairly large overlap threshold to try to maintain overlapping
        # boxes that are still people
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

        # draw the final bounding boxes
        for (xA, yA, xB, yB) in pick:
            cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

        # show some information on the number of bounding boxes
        filename = imagePath[imagePath.rfind("/") + 1:]
        print("[INFO] {}: {} original boxes, {} after suppression".format(
            filename, len(rects), len(pick)))

        # show the output images
        #cv2.imshow("Before NMS", orig)
        plt.figure()
        plt.axis("off")
        plt.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
        #cv2.imshow("After NMS", image)
        plt.figure()
        plt.axis("off")
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        #cv2.waitKey(0)
        cnt = cnt + 1
        # Two images for TEST results
        if cnt == 2:
            break
