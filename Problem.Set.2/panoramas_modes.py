import cv2
import numpy as np
from detectors_descriptors import feature_matching
from blend import blend_modes


def generate_panorama_modes(ordered_img_seq, mode_feature, mode_blend):
    length = len(ordered_img_seq)
    mid = int(length/2)
    i = mid-1
    j = mid+1
    principle_img = ordered_img_seq[mid]
    while(j < length):
        pixels1, pixels2 = feature_matching(
            ordered_img_seq[j], principle_img, mode_feature)
        homo_matrix, _ = cv2.findHomography(pixels1, pixels2,
                                            method=cv2.RANSAC, ransacReprojThreshold=5.0)
        principle_img = blend_modes(
            ordered_img_seq[j], principle_img, homo_matrix, mode_blend)
        principle_img = np.uint8(principle_img)
        j = j+1
    while(i >= 0):
        pixels1, pixels2 = feature_matching(
            ordered_img_seq[i], principle_img, mode_feature)
        homo_matrix, _ = cv2.findHomography(pixels1, pixels2,
                                            method=cv2.RANSAC, ransacReprojThreshold=5.0)
        principle_img = blend_modes(
            ordered_img_seq[i], principle_img, homo_matrix, mode_blend)
        principle_img = np.uint8(principle_img)
        i = i-1
    est_pano = principle_img
    return est_pano


def main():
    img_1 = cv2.imread('inputs/panoramas/grail/grail09.jpg')
    img_2 = cv2.imread('inputs/panoramas/grail/grail10.jpg')
    img_3 = cv2.imread('inputs/panoramas/grail/grail11.jpg')
    img_4 = cv2.imread('inputs/panoramas/grail/grail12.jpg')
    img_5 = cv2.imread('inputs/panoramas/grail/grail13.jpg')
    img_list = []
    img_list.append(img_1)
    img_list.append(img_2)
    img_list.append(img_3)
    img_list.append(img_4)
    img_list.append(img_5)

    # SURF need lower version of Opencv
    modes_feature = ['SIFT','AKAZE','BRISK','ORB']
    for mode in modes_feature:
        pano = generate_panorama_modes(img_list, mode, 'alpha')
        cv2.imwrite(
            f"outputs/panoramas/grail_{mode}.jpg", pano)

    # alpha blending is what I use in main_project
    modes_blend = ['alpha','multiband','poisson','pyramid']
    for mode in modes_blend:
        pano = generate_panorama_modes(img_list, 'SIFT', mode)
        cv2.imwrite(
            f"outputs/panoramas/library_{mode}.jpg", pano)

    # OpenCV use SIFT to stitch
    stitchy = cv2.Stitcher.create()
    (dummy, panorama) = stitchy.stitch(img_list)
    if dummy != cv2.STITCHER_OK:
        print("Failed")
    else:
        print("Suceessful")
        cv2.imwrite(f'outputs/panoramas/grail_cv2.jpg', panorama)


if __name__ == '__main__':
    main()
