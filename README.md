# Panoramic stitching of images

## RGB images

### Methodology
* Detect, extract and match features between a pair of RGB images using feature extractors like SIFT (ORB in this case)
* Use any 4 random matches to calculate the Homography matrix
* Run RANSAC algorithm to remove the outliers in these matched points so that only inliers are used to calculate the Homography matrix
* Warping by multiplying second image by the homography matrix
* Blending images together using their indices.

Note: In between, we would need to calculate the total size of final image. For that, I have used the four corners to get the corners of final image. 

### Dataset
Here is the [drive link](https://drive.google.com/file/d/1sGatCBjhLzxrrQ501NiN0vqGWBr3qyw2/view) for Dataset folder. Can use custom RGB images as well.

## RGB-D images
Let the pair of RGB-D images comprise a source image and a reference image along with their depth images. 
### Methodology
* Detect, extract and match features between source image and reference image.
* Quantize the depth image corresponding to the reference image into m levels where m>10.(For eg: if depth image has values from 0 to 100, then the image quantized to 5 depth levels will only have values 0,20,40,60,80,100)
* Estimate homography matrix for each depth level of the quantized reference depth image. (can directly use function used for RGB images)
* Warp each portion of the reference RGB image corresponding to each depth level using the corresponding homography matrix to obtain an image similar to the source image. (similar to RGB images)


Note: Currently this repo only supports warping the reference image to warp like src image. Blending can be done be similar to RGB images
### Dataset
Here is the [drive link](https://drive.google.com/file/d/14e5UwvNMpWNjgP36_nf5EMCPdsmsV6lP/view) for Dataset folder. Can use custom RGB-D images as well.
