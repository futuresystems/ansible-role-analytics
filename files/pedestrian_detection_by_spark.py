import imutils
from imutils import paths

def apply_batch(imagePath):
        # load the image and resize it to (1) reduce detection time
        # and (2) improve detection accuracy
        import cv2
        from imutils.object_detection import non_max_suppression
        import numpy as np
        import imutils
        # initialize the HOG descriptor/person detector
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        image = cv2.imread(imagePath)
        image = imutils.resize(image, width=min(400, image.shape[1]))
        # detect people in the image
        (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
            padding=(8, 8), scale=1.05)
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
        # draw the final bounding boxes
        for (xA, yA, xB, yB) in pick:
            cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
        return image

mypath = "dataset/INRIAPerson/Test/pos/"
pd = sc.parallelize(paths.list_images(mypath))
pd2 = pd.map(apply_batch)
res = pd2.collect()

import cv2
import matplotlib.pyplot as plt

cnt = 0
for image in res:
    plt.figure()
    plt.axis("off")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    cnt = cnt + 1
    if cnt == 10:
        break
