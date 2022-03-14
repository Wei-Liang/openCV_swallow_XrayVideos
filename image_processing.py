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

if 0:

    for iImg in np.arange(20)*10+1:
        img_name=video_name+'__'+str(iImg)
        equalilzeHistogram(img_name,'CLAHE',clipLimit=4.0, tileGridSize=(32,32))


#raw data (Below) not as good as CLAHE equalized data
# queryImg_name='chin_normal_ori'#'spine_lower_normal_ori'#'chin_normal_ori'#'spine_lower_normal'   # queryImage
# for iImg in np.arange(20)*10+1:
#     img_name=video_name+'__'+str(iImg)
#     siftAndHomo(queryImg_name,img_name)

queryImg_name='spine_lower_normal'#'chin_normal'#'spine_lower_normal'   # queryImage
for iImg in np.arange(20)*10+1:
    img_name=video_name+'__'+str(iImg)+'_equ'
    siftAndHomo(queryImg_name,img_name)


#template matching
#https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html#template-matching

#canny edge detection
#https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html#canny
