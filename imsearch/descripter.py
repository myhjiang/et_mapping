import numpy as np
import cv2
import imutils
import glob
import csv


# color descriptor
class ColorDescriptor:
    def __init__(self, bins):
        # store the number of bins for the 3D histogram
        self.bins = bins

    def describe(self, image):
        # convert the image to the HSV color space and initialize the features used to quantify the image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        features = []

        # grab the dimensions and compute the center of the image
        (h, w) = image.shape[:2]
        (cX, cY) = (int(w * 0.5), int(h * 0.5))

        # divide the image into four rectangles/segments (top-left,  top-right, bottom-right, bottom-left)
        segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h),
                    (0, cX, cY, h)]

        # or maybe two is enough
        # segments = [(0, w, 0, cY), (0, w, cY, h)]

        # loop over the segments
        for (startX, endX, startY, endY) in segments:
            # construct a mask for each segment
            cornerMask = np.zeros(image.shape[:2], dtype="uint8")
            cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)

        # extract a color histogram from the image, then update the feature vector
        hist = self.histogram(image, cornerMask)
        features.extend(hist)
        # return the feature vector
        return features

    def histogram(self, image, mask):
        # extract a 3D color histogram from the masked region of the image
        # using the supplied number of bins per channel
        hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins,
                            [0, 180, 0, 256, 0, 256])
        # normalize histogram, for version 3+
        hist = cv2.normalize(hist, hist).flatten()

        # return the histogram
        return hist
