import numpy as np
from scipy import ndimage, spatial
import cv2
from os import listdir

# h,w = np.shape(img)
# The x-axis follows the w direction
# The y-axis follows the h direction


def gradient_x(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray)
    gray = ndimage.gaussian_filter(gray, 3, mode='reflect')
    grad_x = ndimage.sobel(gray, 1, mode='reflect')
    return grad_x


def gradient_y(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray)
    gray = ndimage.gaussian_filter(gray, 3, mode='reflect')
    grad_y = ndimage.sobel(gray, 0, mode='reflect')
    return grad_y


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


def harris_response(img, alpha, window_size):
    grad_x = gradient_x(img)
    grad_y = gradient_y(img)
    filter = gaussian_blur_kernal_2d(window_size, window_size, 5)
    A = ndimage.convolve(grad_x*grad_x, filter, mode='reflect')
    B = ndimage.convolve(grad_x*grad_y, filter, mode='reflect')
    C = ndimage.convolve(grad_y*grad_y, filter, mode='reflect')
    R = A*C-B**2-alpha*((A+C)**2)
    return R


def corner_selection(R, threshold, min_distance):
    R[R < threshold] = 0
    R = ndimage.maximum_filter(R, size=min_distance)
    h, w = np.shape(R)
    R_selection = np.zeros(np.shape(R))
    d = int((min_distance-1)/2)
    for i in range(d, h-d+1):
        for j in range(d, w-d+1):
            win = np.unique(R[i-d:i+d+1, j-d:j+d+1])
            if (win.size == 1) & (win[0] != 0):
                R_selection[i][j] = R[i][j]
                j = j+d
    pixels = np.argwhere(R_selection > 0)
    pixels = [tuple((p[1], p[0])) for p in pixels]
    return pixels


def histogram_of_gradients(img, pixels):
    grad_x = gradient_x(img)
    grad_y = gradient_y(img)
    grad_dir = np.degrees(np.arctan2(grad_y, grad_x))+22.5
    pad_img = np.pad(img, ((16, 16), (16, 16), (0, 0)), mode='reflect')
    features = []
    for pixel in pixels:
        x = int(pixel[0])-8
        y = int(pixel[1])-8
        H = np.zeros((8))
        for i in range(4):
            for j in range(4):
                H_local, _ = np.histogram(
                    grad_dir[y+4*i:y+4*i+4, x+4*j:x+4*j+4], 8, (-157.5, 202.5), density=False)
                H = H+H_local
        H = H/16
        pos = np.argmax(H)
        grad_prominent = -135+45*pos
        M = cv2.getRotationMatrix2D((16, 16), -grad_prominent, 1.0)
        img_rotate = cv2.warpAffine(
            pad_img[y+8:y+41, x+8:x+41], M, (17, 17))  # (y+8+16)-16:(y+8+16)+17
        grad_x_rotate = gradient_x(img_rotate)
        grad_y_rotate = gradient_y(img_rotate)
        grad_dir_rotate = np.degrees(
            np.arctan2(grad_y_rotate, grad_x_rotate))+22.5
        feature = np.empty((128))
        cnt = 0
        for i in range(4):
            for j in range(4):
                H_local_rotate, _ = np.histogram(
                    grad_dir_rotate[4*i:4*i+4, 4*j:4*j+4], 8, (-157.5, 202.5), density=False)
                feature[cnt*8:cnt*8+8] = H_local_rotate
                cnt = cnt+1
        features.append(feature)
    return features


def feature_matching(img_1, img_2):
    R1 = harris_response(img_1, 0.04, 9)
    R2 = harris_response(img_2, 0.04, 9)
    cor1 = corner_selection(R1, 0.01*np.max(R1), 5)
    cor2 = corner_selection(R2, 0.01*np.max(R1), 5)
    fea1 = histogram_of_gradients(img_1, cor1)
    fea2 = histogram_of_gradients(img_2, cor2)
    dis = spatial.distance.cdist(fea1, fea2, metric='euclidean')
    threshold = 0.6
    pixels_1 = []
    pixels_2 = []
    p1, p2 = np.shape(dis)
    if p1 < p2:
        for p in range(p1):
            dis_min = np.min(dis[p])
            pos = np.argmin(dis[p])
            dis[p][pos] = np.max(dis)
            if dis_min/np.min(dis[p]) <= threshold:
                pixels_1.append(cor1[p])
                pixels_2.append(cor2[pos])
                dis[:, pos] = np.max(dis)

    else:
        for p in range(p2):
            dis_min = np.min(dis[:, p])
            pos = np.argmin(dis[:, p])
            dis[pos][p] = np.max(dis)
            if dis_min/np.min(dis[:, p]) <= threshold:
                pixels_2.append(cor2[p])
                pixels_1.append(cor1[pos])
                dis[pos] = np.max(dis)
    min_len = min(np.shape(cor1)[0], np.shape(cor2)[0])
    rate = np.shape(pixels_1)[0]/min_len
    assert rate >= 0.03, "Fail to Match!"
    return pixels_1, pixels_2


# convert (x,y) to [x,y,1]
def homo_coordinates(pixels):
    pixels = np.asarray(pixels)
    homo_pixels = np.ones((np.shape(pixels)[0], 3))
    homo_pixels[:, 0:2] = pixels
    return homo_pixels


def compute_homography(pixels_1, pixels_2):
    homo_pixels_1 = homo_coordinates(pixels_1)
    homo_pixels_2 = homo_coordinates(pixels_2)
    len = np.shape(pixels_1)[0]
    A = np.zeros((2*len, 9))
    A[0:2*len:2, 0:3] = homo_pixels_1
    A[1:2*len:2, 3:6] = homo_pixels_1
    A[0:2*len:2, 6:9] = -homo_pixels_1*homo_pixels_2[:, 0:1]
    A[1:2*len:2, 6:9] = -homo_pixels_1*homo_pixels_2[:, 1:2]
    U, S, V = np.linalg.svd((np.transpose(A)).dot(A))
    homo_matrix = np.reshape(V[np.argmin(S)], (3, 3))
    return homo_matrix


def align_pair(pixels_1, pixels_2):
    i = 0
    est_homo = np.zeros((3, 3))
    max_inliers = 0
    homo_pixels_1 = homo_coordinates(pixels_1)
    homo_pixels_2 = homo_coordinates(pixels_2)
    len = np.shape(pixels_1)[0]
    threshold = 3
    while i < 567:  # suppose 70% outlier, want 99%
        i = i+1
        rand_indcies = np.random.choice(len, 4, replace=False)
        rand_pixels_1 = [pixels_1[i] for i in rand_indcies]
        rand_pixels_2 = [pixels_2[i] for i in rand_indcies]
        homo_matrix = compute_homography(rand_pixels_1, rand_pixels_2)
        trans_pixels_1 = homo_matrix.dot(np.transpose(homo_pixels_1))
        trans_pixels_1 = trans_pixels_1/trans_pixels_1[2:3, :]
        trans_pixels_1 = np.transpose(trans_pixels_1)
        dis = spatial.distance.cdist(
            trans_pixels_1, homo_pixels_2, metric='euclidean')
        dis = np.diagonal(dis)
        num = np.count_nonzero(dis < threshold)
        if num > max_inliers:
            max_inliers = num
            est_homo = homo_matrix
    return est_homo


def stitch_blend(img_1, img_2, est_homo):
    h1, w1, d1 = np.shape(img_1)  # d=3 RGB
    h2, w2, d2 = np.shape(img_2)
    p1 = est_homo.dot(np.array([0, 0, 1]))
    p2 = est_homo.dot(np.array([0, h1, 1]))
    p3 = est_homo.dot(np.array([w1, 0, 1]))
    p4 = est_homo.dot(np.array([w1, h1, 1]))
    p1 = np.int16(p1/p1[2])
    p2 = np.int16(p2/p2[2])
    p3 = np.int16(p3/p3[2])
    p4 = np.int16(p4/p4[2])
    x_min = min(0, p1[0], p2[0], p3[0], p4[0])
    x_max = max(w2, p1[0], p2[0], p3[0], p4[0])
    y_min = min(0, p1[1], p2[1], p3[1], p4[1])
    y_max = max(h2, p1[1], p2[1], p3[1], p4[1])
    x_range = np.arange(x_min, x_max+1, 1)
    y_range = np.arange(y_min, y_max+1, 1)
    x, y = np.meshgrid(x_range, y_range)
    x = np.float32(x)
    y = np.float32(y)
    homo_inv = np.linalg.pinv(est_homo)
    trans_x = homo_inv[0, 0]*x+homo_inv[0, 1]*y+homo_inv[0, 2]
    trans_y = homo_inv[1, 0]*x+homo_inv[1, 1]*y+homo_inv[1, 2]
    trans_z = homo_inv[2, 0]*x+homo_inv[2, 1]*y+homo_inv[2, 2]
    trans_x = trans_x/trans_z
    trans_y = trans_y/trans_z
    est_img_1 = cv2.remap(img_1, trans_x, trans_y, cv2.INTER_LINEAR)
    est_img_2 = cv2.remap(img_2, x, y, cv2.INTER_LINEAR)
    alpha1 = cv2.remap(np.ones(np.shape(img_1)), trans_x,
                       trans_y, cv2.INTER_LINEAR)
    alpha2 = cv2.remap(np.ones(np.shape(img_2)), x, y, cv2.INTER_LINEAR)
    alpha = alpha1+alpha2
    alpha[alpha == 0] = 2
    alpha1 = alpha1/alpha
    alpha2 = alpha2/alpha
    est_img = est_img_1*alpha1+est_img_2*alpha2
    return est_img


def generate_panorama(ordered_img_seq):
    len = np.shape(ordered_img_seq)[0]
    mid = int(len/2)
    i = mid-1
    j = mid+1
    principle_img = ordered_img_seq[mid]
    while(j < len):
        pixels1, pixels2 = feature_matching(ordered_img_seq[j], principle_img)
        homo_matrix = align_pair(pixels1, pixels2)
        principle_img = stitch_blend(
            ordered_img_seq[j], principle_img, homo_matrix)
        principle_img=np.uint8(principle_img)
        j = j+1  
    while(i >= 0):
        pixels1, pixels2 = feature_matching(ordered_img_seq[i], principle_img)
        homo_matrix = align_pair(pixels1, pixels2)
        principle_img = stitch_blend(
            ordered_img_seq[i], principle_img, homo_matrix)
        principle_img=np.uint8(principle_img)
        i = i-1  
    est_pano = principle_img
    return est_pano

def main():
    img_1 = cv2.imread('inputs/panoramas/parrington/prtn00.jpg')
    img_2 = cv2.imread('inputs/panoramas/parrington/prtn01.jpg')
    img_3 = cv2.imread('inputs/panoramas/parrington/prtn02.jpg')
    img_4 = cv2.imread('inputs/panoramas/parrington/prtn03.jpg')
    img_5 = cv2.imread('inputs/panoramas/parrington/prtn04.jpg')
    img_list=[]
    img_list.append(img_1)
    img_list.append(img_2)
    img_list.append(img_3)
    img_list.append(img_4)
    img_list.append(img_5)
    pano=generate_panorama(img_list)
    cv2.imwrite("outputs/panoramas/1parrington.jpg", pano)

if __name__=='__main__':
    main()

