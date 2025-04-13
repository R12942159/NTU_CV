import numpy as np
import cv2
import random
from tqdm import tqdm
from utils import solve_homography, warping

random.seed(999)

def linear_blending(img1, img2, alpha):
    """
    Linear blending of two images
    :param img1: first image
    :param img2: second image
    :param alpha: blending factor
    :return: blended image
    """
    blended = alpha * img1 + (1 - alpha) * img2
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return blended

def panorama(imgs):
    """
    Image stitching with estimated homograpy between consecutive
    :param imgs: list of images to be stitched
    :return: stitched panorama
    """
    N_POINTS = 100
    ITERATION = 1815
    KEYPOINT4H = 8
    INLIERS_THRESHOLD = 0.3

    h_max = max([x.shape[0] for x in imgs])
    w_max = sum([x.shape[1] for x in imgs])

    # create the final stitched canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]
    last_best_H = np.eye(3)

    # for all images to be stitched:
    for idx in range(len(imgs)-1):
        im2 = imgs[idx]
        im1 = imgs[idx + 1]

        # TODO: 1.feature detection & matching
        orb = cv2.ORB_create()
        keypoint1, descriptor1 = orb.detectAndCompute(im1, None) # queryImage
        keypoint2, descriptor2 = orb.detectAndCompute(im2, None) # trainImage
        # Feature point matching using Brute-Force Matcher
        bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf_matcher.match(descriptor1, descriptor2)
        matches = sorted(matches, key = lambda x: x.distance)[:N_POINTS]
        # Extract the matched keypoints
        query_idx = [match.queryIdx for match in matches]
        train_idx = [match.trainIdx for match in matches]
        # Corresponding actual coordinates
        src_pts = np.array([keypoint1[idx].pt for idx in query_idx]) # img1 feature points coordinates
        dst_pts = np.array([keypoint2[idx].pt for idx in train_idx]) # img2 feature points coordinates

        # TODO: 2. apply RANSAC to choose best H
        max_Inliers = 0
        best_H = np.eye(3)
        for _ in range(ITERATION):
            # Randomly sample a few matches
            rand_idx = random.sample(range(len(src_pts)), KEYPOINT4H)
            src_loc, dst_loc = src_pts[rand_idx], dst_pts[rand_idx]
            H = solve_homography(src_loc, dst_loc)
            # Apply the homography to all points
            U = np.concatenate((src_pts.T, np.ones((1,src_pts.shape[0]))), axis=0) # homogeneous coordinates
            pred_pts = H @ U
            pred_pts = (pred_pts/pred_pts[-1]).T[:,:2]
            # reprojection error
            dst_error = pred_pts - dst_pts
            dst_error = np.linalg.norm(dst_error, axis=1)
            # Count the number of inliers
            inliers = (dst_error < INLIERS_THRESHOLD).sum()
            if inliers > max_Inliers :
                best_H = H.copy()
                max_Inliers = inliers

        # TODO: 3. chain the homographies
        last_best_H = last_best_H @ best_H

        # TODO: 4. apply warping
        dst = warping(im1, dst, last_best_H, 0, h_max, 0, w_max, direction='b')

        # # Linear Blending
        # im1_resize = cv2.resize(im1, (dst.shape[1], dst.shape[0]))
        # im2_resize = cv2.resize(im2, (dst.shape[1], dst.shape[0]))
        # alpha = np.where(dst == 0, 0, 1)
        # dst = linear_blending(im1_resize, im2_resize, alpha)

    return dst 

if __name__ == "__main__":
    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)