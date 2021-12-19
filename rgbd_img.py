import cv2
from collections import Counter
import numpy as np
from utils import ransac, keypoints_matcher
from scipy.ndimage import gaussian_filter


def quantize_depth(depth):
    # I have used non uniform quantization since most of the points were present in depth level 0-100 with very few above 150 


    # anything above 180 goes to 200
    depth[depth>=180] = 200

    # anything between 100-200, is quantized in levels of 20 
    depth[(100<=depth)*(depth<180)] = (depth[(100<=depth)*(depth<180)]//20)*20 + 10

    # anything less than 100 is quantized in levels of 10
    depth[depth<100] = (depth[depth<100]//10)*10 + 5

    return depth

def find_pts(kp_src, kp_ref, m12, depth_ref, quantized_list):
    """
    extracting the matched points and segregating them based on the depth level

    """
    ref_imgs_ = [[] for a in range(len(quantized_list))]
    src_imgs_ = [[] for a in range(len(quantized_list))]

    for m in m12:
        x1, y1 = kp_src[m.queryIdx].pt
        x2, y2 = kp_ref[m.trainIdx].pt
        
        depth = depth_ref[int(y2), int(x2)]
        idx = np.where(quantized_list==depth)[0][0]

        src_imgs_[idx].append([x1, y1])
        ref_imgs_[idx].append([x2, y2])
    
    for i in range(len(ref_imgs_)):
        ref_imgs_[i] = np.float32(ref_imgs_[i])
        src_imgs_[i] = np.float32(src_imgs_[i])
    
    return src_imgs_, ref_imgs_

def get_H_list(src_list, ref_list, quantized_list, img_size, built_in):
    H_list = []
    for a in range(len(quantized_list)):
        if len(src_list[a]) < 4:
            H_list.append([])
            continue
        if built_in:
            (curr_H, _) = cv2.findHomography(src_list[a], ref_list[a], cv2.RANSAC, 5.0)
        else:
            curr_H = ransac(src_list[a], ref_list[a], img_size, 500, 5)
        H_list.append(curr_H)
    
    inv_H_list = []
    for h in H_list:
        if len(h)==0:
            inv_H_list.append([])
            continue
        inv_H_list.append(np.linalg.inv(h))
    
    return H_list, inv_H_list

# function warp1, warp2 and warp3 explaing in report

def warp1(src_img, ref_img, ref_depth, H_list, quantized_list):
    ref_final = np.zeros(src_img.shape)
    for y_src in range(ref_final.shape[0]):
        for x_src in range(ref_final.shape[1]):
            curr_pt_src =  np.float64([x_src, y_src, 1.]).reshape(3, 1)
            
            
            depth = ref_depth[y_src, x_src]

            idx = np.where(quantized_list==depth)[0][0]

            if len(H_list[idx])==0:
                continue
            pt2 = np.matmul(H_list[idx], curr_pt_src).reshape(3, )
            pt2 = pt2/pt2[2]
            x_ref, y_ref = [int(round(pt2[i])) for i in range(2)]
            if 0 <= x_ref < ref_img.shape[1] and 0 <= y_ref < ref_img.shape[0]:
                ref_final[y_src, x_src] = ref_img[y_ref, x_ref]
    return ref_final

def get_depth(final_depth, x, y):
    hf, wf = final_depth.shape[:2]

    cnt = Counter()

    # offset for 8 points around the current point 
    dx = [1, 1, 1, -1, -1, -1, 0, 0]
    dy = [1, -1, 0, 1, -1, 0, 1, -1]
    
    # can change a to increase the size of square - currently 1
    for a in range(1, 2):
        for b in range(8):
            curr_x, curr_y = x+a*dx[b], y+a*dy[b]
            try:
                curr_dep = final_depth[curr_y, curr_x]
                if curr_dep!=0:
                    cnt[curr_dep]+=1
            except IndexError:
                # on the edges and corners
                continue
    

    # if there are no nearby point mapped
    if len(cnt.keys())==0:
        return -1
    
    # returns values that occured most of the times, also add that to final depth array
    dep_final = list(cnt.keys())[0]
    final_depth[curr_y, curr_x] = dep_final
    return dep_final

def warp2(src_img, src_depth, ref_img, ref_depth, H_list, inv_H_list, quantized_list):
    ref_final = np.zeros(src_img.shape)
    depth_final = np.zeros(src_depth.shape)

    # Looping over the reference image and mapping them to source image
    for y_ref in range(ref_img.shape[0]):
        for x_ref in range(ref_img.shape[1]):
            curr_pt_ref = np.float64([x_ref, y_ref, 1]).reshape(3, 1)

            depth = ref_depth[y_ref, x_ref]
            idx = np.where(quantized_list==depth)[0][0]

            if len(inv_H_list[idx])==0:
                continue
            
            pt2 = np.matmul(inv_H_list[idx], curr_pt_ref).reshape(3, )
            pt2 /= pt2[2]
            x_src, y_src = [int(round(pt2[i])) for i in range(2)]
            if 0 <= x_src < src_img.shape[1] and 0 <= y_src < src_img.shape[0]:
                ref_final[y_src, x_src] = ref_img[y_ref, x_ref]
                depth_final[y_src, x_src] = ref_depth[y_ref, x_ref]


    # To interpolate the reference mapped image, checking the nearby pixels and using H matrix used by most of them
    for y_src in range(ref_final.shape[0]):
        for x_src in range(ref_final.shape[1]):

            # if the pixel is already mapped, leaving those
            if ref_final[y_src,x_src].any():
                continue
            
            curr_pt_src =  np.float64([x_src, y_src, 1.]).reshape(3, 1)
            depth = get_depth(depth_final, x_src, y_src)
            if depth==-1:
                continue
            
            idx = np.where(quantized_list==depth)[0][0]
            
            if len(H_list[idx])==0:
                continue
            pt2 = np.matmul(H_list[idx], curr_pt_src).reshape(3, )
            pt2 = pt2/pt2[2]
            x_ref, y_ref = [int(round(pt2[i])) for i in range(2)]
            if 0 <= x_ref < ref_img.shape[1] and 0 <= y_ref < ref_img.shape[0]:
                ref_final[y_src, x_src] = ref_img[y_ref, x_ref]

    return ref_final

def warp3(src_img, ref_img, ref_depth, H_list, quantized_list):
    ref_final = np.zeros(src_img.shape)

    ans=0
    for y_src in range(ref_final.shape[0]):
        for x_src in range(ref_final.shape[1]):
            curr_pt_src =  np.float64([x_src, y_src, 1.]).reshape(3, 1)

            # MAP depth using rev_H
            for H_idx in range(len(H_list)):
                curr_H = H_list[H_idx]
                if len(curr_H)==0:
                    continue
                
                pt_ref = np.matmul(curr_H, curr_pt_src).reshape(3, )
                pt_ref = pt_ref/pt_ref[2]
                x_ref, y_ref = [int(round(pt_ref[i])) for i in range(2)]
                
                if 0 <= x_ref < ref_img.shape[1] and 0 <= y_ref < ref_img.shape[0]:
                    depth = ref_depth[y_ref, x_ref]
                    depth_idx = np.where(quantized_list==depth)[0][0]
                    if depth_idx == H_idx:
                        ans+=1
                        ref_final[y_src, x_src] = ref_img[y_ref, x_ref]
                        break
    return ref_final

def stitch(src_img, ref_img, src_depth, ref_depth, built_in=False, **kwargs):
    """
    Although this function is named as stitch, blending has not been done. 
    """

    # quantizing
    ref_depth = quantize_depth(ref_depth)
    src_depth = quantize_depth(src_depth)
    quantized_list = np.unique(ref_depth)

    # feature matching
    descriptor = cv2.ORB_create(nfeatures=20000)
    (kp1, f1) = descriptor.detectAndCompute(src_img, None)
    (kp2, f2) = descriptor.detectAndCompute(ref_img, None)
    img_size = [src_img.shape, ref_img.shape]
    m12 = keypoints_matcher(f1, f2)

    src_list, ref_list = find_pts(kp1, kp2, m12, ref_depth, quantized_list)
    H_list, _ = get_H_list(src_list, ref_list, quantized_list, img_size, built_in)

    # Other possible ways of warping
    # ref_final1 = warp1(src_img, ref_img, ref_depth, H_list, quantized_list)
    # ref_final1_sr = warp1(src_img, ref_img, src_depth, H_list, quantized_list)
    # ref_final2 = warp2(src_img, src_depth, ref_img, ref_depth, H_list, inv_H_list, quantized_list)
    ref_final3 = warp3(src_img, ref_img, ref_depth, H_list, quantized_list)
    ref_final3 = np.asarray(ref_final3, dtype=np.uint8)
    ref_final3 = cv2.medianBlur(ref_final3, 3)
    return ref_final3