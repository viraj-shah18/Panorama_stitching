import cv2
import numpy as np
from tqdm import tqdm
from utils import ransac, keypoints_matcher

# finding the points
def find_pts(kp1, kp2, m12):
    """
    extracting the matched points

    """
    img1_ = []
    img2_ = []

    for m in m12:
        x1, y1 = kp1[m.queryIdx].pt
        x2, y2 = kp2[m.trainIdx].pt
        img1_.append([x1, y1])
        img2_.append([x2, y2])
    return np.float32(img1_), np.float32(img2_)


def warp_blend(image1, image2, H):
    """
    Parameter H should be for going from image2 to image1
    i.e image1 = H * image2

    """
    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]

    # finding the min and max of height and width of the final image
    # Just doing 4 corners as that would give outermost edges of the final image
    minH = -1
    maxH = height2
    minW = -1
    maxW = width2

    corner_x = [0, 0, width2 - 1, width2 - 1]
    corner_y = [0, height2 - 1, 0, height2 - 1]

    for a in range(4):
        curr_pt = np.array([corner_x[a], corner_y[a], 1]).reshape(3, 1)
        pp = np.matmul(H, curr_pt).T.reshape(
            3,
        )
        pp = pp / pp[2]
        xf, yf = [int(round(pp[i])) for i in range(2)]

        if xf < minW:
            minW = xf
        if xf > maxW:
            maxW = xf
        if yf < minH:
            minH = yf
        if yf > maxH:
            maxH = yf

    minH = max(minH, -500)
    maxH = min(1500, maxH)

    minW = max(minW, -500)
    maxW = min(2500, maxW)

    # image 3 is the final image. starting with zeros and
    # finding the corresponding match using the H matrix

    img3 = np.zeros((maxH - minH, maxW - minW, 3))
    img3[-minH : height1 - minH, -minW : width1 - minW] = image1
    inv_H = np.linalg.inv(H)

    for h in tqdm(range(maxH - minH), desc="Warping Images"):
        for w in range(maxW - minW):
            curr_pt = np.float64([w + minW, h + minH, 1.0]).reshape(3, 1)
            pt2 = np.matmul(inv_H, curr_pt).reshape(
                3,
            )
            pt2 = pt2 / pt2[2]
            xf, yf = [int(round(pt2[i])) for i in range(2)]
            if 0 <= xf < width2 and 0 <= yf < height2:
                img3[h, w] = image2[yf, xf]
    return np.uint8(img3)


def warp_blend2(image1, image2, H):
    """
    Parameter H should be for going from image2 to image1
    i.e image1 = H * image2

    """
    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]

    # finding the min and max of height and width of the final image
    # Just doing 4 corners as that would give outermost edges of the final image
    minH = 0
    maxH = height2
    minW = 0
    maxW = width2

    corner_x = [0, 0, width2 - 1, width2 - 1]
    corner_y = [0, height2 - 1, 0, height2 - 1]

    for a in range(4):
        curr_pt = np.array([corner_x[a], corner_y[a], 1]).reshape(3, 1)
        pp = np.matmul(H, curr_pt).T.reshape(
            3,
        )
        pp = pp / pp[2]
        xf, yf = [int(round(pp[i])) for i in range(2)]

        if xf < minW:
            minW = xf
        if xf > maxW:
            maxW = xf
        if yf < minH:
            minH = yf
        if yf > maxH:
            maxH = yf

    # image 3 is the final image. starting with zeros and
    # finding the corresponding match using the H matrix
    img3 = np.zeros((maxH - minH, maxW - minW, 3))
    img3[-minH : height1 - minH, -minW : width1 - minW] = image1

    for h in tqdm(range(height2), desc="Warping images-P1"):
        for w in range(width2):
            curr_pt = np.float64([w, h, 1.0]).reshape(3, 1)
            pt2 = np.matmul(H, curr_pt).reshape(
                3,
            )
            pt2 = pt2 / pt2[2]
            xf, yf = [int(round(pt2[i])) for i in range(2)]
            # if 0 <= xf < width2 and 0 <= yf < height2:
            try:
                img3[yf - minH, xf - minW] = image2[h, w]
            except IndexError:
                continue

    inv_H = np.linalg.inv(H)

    for h in tqdm(range(maxH - minH), desc="Warping images-P2"):
        for w in range(maxW - minW):
            if img3[h, w].any():
                continue
            curr_pt = np.float64([w + minW, h + minH, 1.0]).reshape(3, 1)
            pt2 = np.matmul(inv_H, curr_pt).reshape(
                3,
            )
            pt2 = pt2 / pt2[2]
            xf, yf = [int(round(pt2[i])) for i in range(2)]
            if 0 <= xf < width2 and 0 <= yf < height2:
                img3[h, w] = image2[yf, xf]
    return np.uint8(img3)


def stitch(img1, img2, max_iter=400, threshold=5.0, **kwargs):
    """
    All the functions to create a main stiching function
    """
    descriptor = cv2.ORB_create(nfeatures=10000)
    (kp1, f1) = descriptor.detectAndCompute(img1, None)
    (kp2, f2) = descriptor.detectAndCompute(img2, None)
    img_size = [img1.shape, img2.shape]
    m12 = keypoints_matcher(f1, f2)
    # print(len(m12))
    # show_matches(img1, kp1, img2, kp2, m12)
    ipts1, ipts2 = find_pts(kp1, kp2, m12)
    H = ransac(ipts1, ipts2, img_size, max_iter, threshold)
    img3 = warp_blend2(img1, img2, np.linalg.inv(H))
    return img3
