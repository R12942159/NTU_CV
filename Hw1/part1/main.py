import numpy as np
import cv2
import argparse
from DoG import Difference_of_Gaussian


def plot_keypoints(img_gray, keypoints, save_path):
    img = np.repeat(np.expand_dims(img_gray, axis = 2), 3, axis = 2)
    for y, x in keypoints:
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
    cv2.imwrite(save_path, img)

def main():
    parser = argparse.ArgumentParser(description='main function of Difference of Gaussian')
    parser.add_argument('--threshold', default=5.0, type=float, help='threshold value for feature selection')
    parser.add_argument('--image_path', default='./testdata/1.png', help='path to input image')
    args = parser.parse_args()

    print('Processing %s ...'%args.image_path)
    img = cv2.imread(args.image_path, 0).astype(np.float32)

    ### TODO ###
    # create DoG class
    DoG = Difference_of_Gaussian(args.threshold)
    
    # find keypoint from DoG and sort it
    keypoints = DoG.get_keypoints(img)

    # 將偵測到的關鍵點繪製在影像上並存檔
    output_path = args.image_path.replace('.png', f'_{args.threshold}threshold_kpts.png')
    plot_keypoints(img, keypoints, output_path)


if __name__ == '__main__':
    main()