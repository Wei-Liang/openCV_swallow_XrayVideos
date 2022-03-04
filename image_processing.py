import numpy as np
from img_processing_lib import *
import cv2
from matplotlib import pyplot as plt
video_name='mixed - Normal'
#mixed - Neuro
segmentVideoIntoImages=0


if segmentVideoIntoImages:
    sec = 0
    frameRate = 30
    count=1

    success = getFrame(video_name,sec,count)
    while success:
        count = count + 1
        sec = sec + 1/frameRate
        sec = round(sec, 2)
        success = getFrame(video_name,sec,count)

if 0:
    img_name=video_name+'__172'
    compareHistogramEqualizers(img_name)


for iImg in np.arange(20)*10+1:
    img_name=video_name+'__'+str(iImg)
    equalilzeHistogram(img_name,'CLAHE',clipLimit=4.0, tileGridSize=(32,32))


queryImg_name='chin_normal'#'spine_lower_normal'   # queryImage
for iImg in np.arange(20)*10+1:
    img_name=video_name+'__'+str(iImg)+'_equ'
    siftAndHomo(queryImg_name,img_name)

    # # Initiate SIFT detector
    # img1 = cv2.imread('../data/'+queryImg_name'.jpg',0) 
    # img2 = cv2.imread('../data/'+img_name+'_equ.jpg',0) # trainImage
    # sift = cv2.SIFT_create()
    # MIN_MATCH_COUNT = 10

    # # find the keypoints and descriptors with SIFT
    # kp1, des1 = sift.detectAndCompute(img1,None)
    # kp2, des2 = sift.detectAndCompute(img2,None)

    # FLANN_INDEX_KDTREE = 0
    # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    # search_params = dict(checks = 50)

    # flann = cv2.FlannBasedMatcher(index_params, search_params)

    # matches = flann.knnMatch(des1,des2,k=2)

    # # store all the good matches as per Lowe's ratio test.
    # good = []
    # for m,n in matches:
    #     if m.distance < 0.7*n.distance:
    #         good.append(m)

    # if len(good)>MIN_MATCH_COUNT:
    #     src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    #     dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    #     M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    #     matchesMask = mask.ravel().tolist()

    #     h,w = img1.shape
    #     pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    #     dst = cv2.perspectiveTransform(pts,M)

    #     img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    # else:
    #     print('Not enough matches are found - %d/%d' % (len(good),MIN_MATCH_COUNT))
    #     matchesMask = None


    # draw_params = dict(matchColor = (0,255,0), # draw matches in green color
    #                    singlePointColor = None,
    #                    matchesMask = matchesMask, # draw only inliers
    #                    flags = 2)


    # img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

    # plt.figure()
    # plt.imshow(img3, 'gray')
    # plt.savefig('../results/'+img_name+'_'+queryImg_name+'_Detection.png', dpi=300)

#template matching
#https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html#template-matching

#canny edge detection
#https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html#canny
