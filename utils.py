import numpy as np
from tqdm import tqdm
import cv2

# Keypoints matching
def keypoints_matcher(f1, f2):
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = matcher.knnMatch(f1, f2, 2)
    final_matches = []
    RATIO = 0.7
    for (m1, m2) in raw_matches:
        if m1.distance < m2.distance * RATIO:
            final_matches.append(m1)
    return final_matches


def show_matches(img1, kp1, img2, kp2, matches):
    img12 = cv2.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        np.random.choice(matches, 20),
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imshow(img12)


def find_T_matrix(img_size):
    """
    Normalising factor used
    """
    (m, n, _) = img_size
    T = np.zeros(shape=(3, 3))
    T[0, 0] = 2 / n
    T[1, 1] = 2 / m
    T[2, 2] = 1
    T[0, 2] = -n / 2
    T[1, 2] = -m / 2
    return T


def get_random_pts(img1_pts, img2_pts):
    # would always require 4 points to determine the solution, using the null space
    pts1 = []
    pts2 = []

    # using random can give out same points, removing them
    used_pts = set()

    for a in range(4):
        random_no = np.random.randint(low=0, high=img1_pts.shape[0])

        # running the loop till we get unused point
        while random_no in used_pts:
            random_no = np.random.randint(low=0, high=img1_pts.shape[0])
        used_pts.add(random_no)

        r_pt1 = img1_pts[random_no]
        r_pt2 = img2_pts[random_no]
        pts1.append(r_pt1)
        pts2.append(r_pt2)

    return np.array(pts1), np.array(pts2)


# Creating a homography matrix
def get_norm_H(img1_pts, img2_pts, img_sizes):
    A = np.zeros((2 * img1_pts.shape[0], 9))

    T1 = find_T_matrix(img_sizes[0])
    T2 = find_T_matrix(img_sizes[1])
    for pt in range(img1_pts.shape[0]):

        p1 = np.array([img1_pts[pt][0], img1_pts[pt][1], 1])
        p2 = np.array([img2_pts[pt][0], img2_pts[pt][1], 1])

        (x1, y1, w1) = np.matmul(T1, p1)
        (x2, y2, w2) = np.matmul(T2, p2)

        first_row = np.array(
            [0, 0, 0, -w2 * x1, -w2 * y1, -w2 * w1, y2 * x1, y2 * y1, y2 * w1]
        )

        second_row = np.array(
            [w2 * x1, w2 * y1, w2 * w1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2 * w1]
        )

        # A.append(first_row)
        A[2 * pt, :] = first_row
        A[2 * pt + 1, :] = second_row
        # A.append(second_row)

    # A = np.matrix(A)
    u, sing, v = np.linalg.svd(A)
    last_row = v[-1].copy()

    h_own_unnorm = last_row.reshape((3, 3))

    h_own_norm = np.matmul(np.matmul(np.linalg.inv(T2), h_own_unnorm), T1)
    h_own_norm = h_own_norm / h_own_norm[-1, -1]
    return h_own_norm


# RANSAC
def ransac(img1_pts, img2_pts, img_size, num_iter=500, threshold=5):
    # according to the paper, num of iter must be 500 to reduce the probability of missing inlier to 1e-14
    curr_max = -1

    # storing all the inlier pts
    max_inpts1 = []
    max_inpts2 = []

    for a in tqdm(range(num_iter), desc="Running RANSAC"):
        pts1, pts2 = get_random_pts(img1_pts, img2_pts)
        h = get_norm_H(pts1, pts2, img_size)
        # h, _ = cv2.findHomography(pts1.reshape(-1, 1, 2), pts2.reshape(-1, 1, 2), 0)
        inliers = 0
        curr_pts1 = []
        curr_pts2 = []

        for a in range(img1_pts.shape[0]):

            # transforming to projective geometry co-ordinates
            original_point = img1_pts[a]
            original_point = np.append(original_point, 1)
            original_point = np.float64(original_point)

            # using the current H matrix to estimate img2-point
            img2_estimate = np.matmul(h, original_point)

            # print(img2_pts[a].shape, img2_estimate.shape)
            # converting its 3rd co-ordinate to 1 and extracting (x, y)
            img2_estimate = (img2_estimate / img2_estimate[-1])[0:2]

            # comparing with threshold
            if np.linalg.norm(img2_pts[a] - img2_estimate, 2) < threshold:
                inliers += 1
                curr_pts1.append(img1_pts[a])
                curr_pts2.append(img2_pts[a])

        if inliers > curr_max:
            curr_max = inliers
            max_inpts1 = np.array(curr_pts1)
            max_inpts2 = np.array(curr_pts2)

    if type(max_inpts1) == type([]):
        print("Empty recieved, change parameters")
        return
    final_H = get_norm_H(max_inpts1, max_inpts2, img_size)
    # print(inliers)
    # final_H, _ = cv2.findHomography(max_inpts1.reshape(-1, 1, 2), max_inpts2.reshape(-1, 1, 2), 0)
    return final_H
