import cv2
from os import listdir
paths=listdir('inputs/panoramas')
for path in paths:
    img_list=[]
    imgs_path=listdir(f'inputs/panoramas/{path}')
    maximum=7
    for img_path in imgs_path:
        if maximum==0:
            break
        img=cv2.imread(f'inputs/panoramas/{path}/{img_path}')
        img_list.append(img)
        maximum=maximum-1
    stitchy=cv2.Stitcher.create()
    (dummy,panorama)=stitchy.stitch(img_list)
    if dummy != cv2.STITCHER_OK:
        print("Failed")
    else:
        print("Suceessful")
        cv2.imwrite(f'outputs/panoramas/{path}_cv2.jpg',panorama)
