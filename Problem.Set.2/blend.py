import numpy as np
import cv2


def blend_modes(img_1, img_2, est_homo, mode):
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
    est_img = np.zeros(np.shape(est_img_1))

    if mode == 'alpha':
        alpha1 = cv2.remap(np.ones(np.shape(img_1)), trans_x,
                           trans_y, cv2.INTER_LINEAR)
        alpha2 = cv2.remap(np.ones(np.shape(img_2)), x, y, cv2.INTER_LINEAR)
        alpha = alpha1+alpha2
        alpha[alpha == 0] = 2
        alpha1 = alpha1/alpha
        alpha2 = alpha2/alpha
        est_img = est_img_1*alpha1+est_img_2*alpha2

    else:
        x1 = [p[1] for p in np.argwhere(est_img_1 > 0)]
        right_1 = np.max(x1)
        left_1 = np.min(x1)
        x2 = [p[1] for p in np.argwhere(est_img_2 > 0)]
        right_2 = np.max(x2)
        left_2 = np.min(x2)
        if left_1 > left_2:
            tmp = est_img_1
            est_img_1 = est_img_2
            est_img_2 = tmp
            tmp_r = right_1
            right_1 = right_2
            right_2 = tmp_r
            tmp_l = left_1
            left_1 = left_2
            left_2 = tmp_l
        # now, est_img_1 is the left side

        if mode == 'multiband':
            len = right_1-left_2
            est_img[:, :left_2+1, :] = est_img_1[:, :left_2+1, :]
            est_img[:, right_1:, :] = est_img_2[:, right_1:, :]
            for i in range(left_2+1, right_1):
                est_img[:, i, :] = (1-(i-left_2)/len)*est_img_1[:,i, :]+(1-(right_1-i)/len)*est_img_2[:, i, :]

        elif mode == 'poisson':
            est_img[:, :right_1+1, :] = est_img_1[:, :right_1+1, :]
            est_img[:, right_1:, :] = est_img_2[:, right_1:, :]
            y1 = [p[0]
                  for p in np.argwhere(est_img_1[:, left_2+1:right_1, :] > 0)]
            y2 = [p[0]
                  for p in np.argwhere(est_img_2[:, left_2+1:right_1, :] > 0)]
            up = max(np.max(y1), np.max(y2))
            down = min(np.min(y1), np.min(y2))
            src = np.uint8(est_img_1[down+1:up, left_2:right_1+1, :])
            mask = np.ones(np.shape(src), dtype=np.uint8)
            mask = 255*mask
            dst = np.uint8(est_img)
            pos = (int((left_2+1+right_1)/2), int((up+down)/2))
            flag = cv2.MIXED_CLONE
            est_img = cv2.seamlessClone(src, dst, mask, pos, flag)

        elif mode == 'pyramid':
            est_img = np.hstack(
                (est_img_1[:, :right_1], est_img_2[:, left_2:]))
            layer = est_img_1.copy()
            gaussian_pyramid = [layer]
            for i in range(3):
                layer = cv2.pyrDown(layer)
                gaussian_pyramid.append(layer)
            layer = gaussian_pyramid[2]
            laplacian_pyramid = [layer]
            for i in range(2, 0, -1):
                size = (gaussian_pyramid[i - 1].shape[1],
                        gaussian_pyramid[i - 1].shape[0])
                gaussian_expanded = cv2.pyrUp(
                    gaussian_pyramid[i], dstsize=size)
                laplacian = cv2.subtract(
                    gaussian_pyramid[i - 1], gaussian_expanded)
                laplacian_pyramid.append(laplacian)
            layer = est_img_2.copy()
            gaussian_pyramid2 = [layer]
            for i in range(3):
                layer = cv2.pyrDown(layer)
                gaussian_pyramid2.append(layer)
            layer = gaussian_pyramid2[2]
            laplacian_pyramid2 = [layer]
            for i in range(2, 0, -1):
                size = (gaussian_pyramid2[i - 1].shape[1],
                        gaussian_pyramid2[i - 1].shape[0])
                gaussian_expanded = cv2.pyrUp(
                    gaussian_pyramid2[i], dstsize=size)
                laplacian = cv2.subtract(
                    gaussian_pyramid2[i - 1], gaussian_expanded)
                laplacian_pyramid2.append(laplacian)
            est_img_pyramid = []
            for img1_lap, img2_lap in zip(laplacian_pyramid, laplacian_pyramid2):
                cols, rows, ch = img1_lap.shape
                laplacian = np.hstack(
                    (img1_lap[:, 0:int(cols/2)], img2_lap[:, int(cols/2):]))
                est_img_pyramid.append(laplacian)
            est_img_reconstructed = est_img_pyramid[0]
            for i in range(1, 3):
                size = (est_img_pyramid[i].shape[1],
                        est_img_pyramid[i].shape[0])
                est_img_reconstructed = cv2.pyrUp(
                    est_img_reconstructed, dstsize=size)
                est_img_reconstructed = cv2.add(
                    est_img_pyramid[i], est_img_reconstructed)
            est_img = est_img_reconstructed
    return est_img
