import numpy as np
import cv2
import math
from os import listdir


def cross_correlation_2d(filter, img):
    d = np.shape(filter)
    ph = int((d[0]-1)/2)
    pw = int((d[1]-1)/2)
    h, w = img.shape
    pad_img = np.zeros((h+2*ph, w+2*pw))
    pad_img[ph:ph+h, pw:pw+w] = img
    filter = filter.reshape(d[0]*d[1], 1)
    for i in range(h):
        for j in range(w):
            img[i][j] = np.dot(
                pad_img[i:i+d[0], j:j+d[1]].reshape(1, d[0]*d[1]), filter)
            if(img[i][j] < 0):
                img[i][j] = 0
            elif(img[i][j] > 255):
                img[i][j] = 255
    return img


def convolve_2(filter, img):
    h, w = filter.shape
    trans_filter_1 = np.zeros((h, w))
    trans_filter_2 = np.zeros((h, w))
    for i in range(h):
        trans_filter_1[i] = filter[h-i-1]
    for j in range(w):
        trans_filter_2[:, j] = trans_filter_1[:, w-j-1]
    return cross_correlation_2d(trans_filter_2, img)


def gaussian_blur_kernal_2d(height, width, theta):
    kernel = np.ones((height, width))
    kernel = kernel*1/(2*np.pi*theta)
    oi = int((height-1)/2)
    oj = int((width-1)/2)
    for i in range(height):
        kernel[i] = kernel[i] * np.e**(-((i-oi)**2)/(2*theta**2))
    for j in range(width):
        kernel[:, j] = kernel[:, j] * np.e**(-((j-oj)**2)/(2*theta**2))
    return kernel/(np.sum(kernel))


def low_pass(img, sigma, kernel_size):
    kernel_1 = gaussian_blur_kernal_2d(1, kernel_size, sigma)
    kernel_2 = gaussian_blur_kernal_2d(kernel_size, 1, sigma)
    img = convolve_2(kernel_1, img)
    img = convolve_2(kernel_2, img)
    return img


def high_pass(img, sigma, kernel_size):
    copy = np.copy(img)
    kernel_1 = gaussian_blur_kernal_2d(1, kernel_size, sigma)
    kernel_2 = gaussian_blur_kernal_2d(kernel_size, 1, sigma)
    img = convolve_2(kernel_1, img)
    img = convolve_2(kernel_2, img)
    return copy-img


def hybrid_image(high_img, low_img, sigma_high, sigma_low, size_high, size_low):
    high_img = high_pass(high_img, sigma_high, size_high)
    low_img = low_pass(low_img, sigma_low, size_low)
    result = high_img+low_img
    return result


def run_hybrid_img(sigma_high=13, sigma_low=21, img_num=5):
    files = listdir()
    path_left = []
    path_right = []
    for i in range(1, img_num+1):
        for file in files:
            if("left_"+str(i) in file):
                path_left.append(file)
            elif("right_"+str(i) in file):
                path_right.append(file)
    for i in range(img_num):
        left_img = cv2.imread(f'./imgs/{path_left[i]}')
        right_img = cv2.imread(f'./imgs/{path_right[i]}')
        h, w, d = np.shape(left_img)
        size_high = int(math.log(min(h, w)))
        if(size_high % 2 == 0):
            size_high = size_high-1
        if(size_high < 5):
            size_high = 5
        size_low = 2*size_high+1
        print(f'hybrid_{i+1}.png Processing...')
        result = np.empty((h, w, d))
        result[:, :, 0] = hybrid_image(
            left_img[:, :, 0], right_img[:, :, 0], sigma_high, sigma_low, size_high, size_low)
        result[:, :, 1] = hybrid_image(
            left_img[:, :, 1], right_img[:, :, 1], sigma_high, sigma_low, size_high, size_low)
        result[:, :, 2] = hybrid_image(
            left_img[:, :, 2], right_img[:, :, 2], sigma_high, sigma_low, size_high, size_low)
        cv2.imwrite(f'./imgs/hybrid_{i+1}.png', result)
        print(f'hybrid_{i+1}.png Done!')


run_hybrid_img()

