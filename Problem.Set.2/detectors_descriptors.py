import cv2
import numpy as np


def feature_SIFT(img1, img2):
    sift = cv2.SIFT_create()
    keyPoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keyPoints2, descriptors2 = sift.detectAndCompute(img2, None)
    return keyPoints1, descriptors1, keyPoints2, descriptors2


def feature_SURF(img1, img2):
    surf = cv2.xfeatures2d.SURF_create(400)
    keyPoints1, descriptors1 = surf.detectAndCompute(img1, None)
    keyPoints2, descriptors2 = surf.detectAndCompute(img2, None)
    return keyPoints1, descriptors1, keyPoints2, descriptors2


def feature_AKAZE(img1, img2):
    akaze = cv2.AKAZE_create()
    keyPoints1, descriptors1 = akaze.detectAndCompute(img1, None)
    keyPoints2, descriptors2 = akaze.detectAndCompute(img2, None)
    return keyPoints1, descriptors1, keyPoints2, descriptors2


def feature_BRISK(img1, img2):
    brisk = cv2.BRISK_create()
    keyPoints1, descriptors1 = brisk.detectAndCompute(img1, None)
    keyPoints2, descriptors2 = brisk.detectAndCompute(img2, None)
    return keyPoints1, descriptors1, keyPoints2, descriptors2


def feature_ORB(img1, img2):
    orb = cv2.ORB_create()
    keyPoints1 = orb.detect(img1, None)
    keyPoints1, descriptors1 = orb.compute(img1, keyPoints1)
    keyPoints2 = orb.detect(img2, None)
    keyPoints2, descriptors2 = orb.compute(img2, keyPoints2)
    return keyPoints1, descriptors1, keyPoints2, descriptors2


def feature_matching(img1, img2, mode):
    if mode == 'SIFT':
        keyPoints1, descriptors1, keyPoints2, descriptors2 = feature_SIFT(
            img1, img2)
        bf = cv2.BFMatcher()
    elif mode == 'SURF':
        keyPoints1, descriptors1, keyPoints2, descriptors2 = feature_SURF(
            img1, img2)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    elif mode == 'BRISK':
        keyPoints1, descriptors1, keyPoints2, descriptors2 = feature_BRISK(
            img1, img2)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    elif mode == 'AKAZE':
        keyPoints1, descriptors1, keyPoints2, descriptors2 = feature_AKAZE(
            img1, img2)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    elif mode == 'ORB':
        keyPoints1, descriptors1, keyPoints2, descriptors2 = feature_ORB(
            img1, img2)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    pixels_1 = np.int16(
        [keyPoints1[m.queryIdx].pt for m in matches]).reshape(-1,  2)
    pixels_2 = np.int16(
        [keyPoints2[m.trainIdx].pt for m in matches]).reshape(-1,  2)
    return pixels_1, pixels_2


