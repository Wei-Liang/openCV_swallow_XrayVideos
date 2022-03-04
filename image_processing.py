import numpy as np
from img_processing_lib import *
import cv2
from matplotlib import pyplot as plt
video_name='mixed - Neuro'
#mixed - Neuro
segmentVideoIntoImages=1


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


img_name=video_name+'__172'
img=cv2.imread('../data/'+img_name+'.jpg',0)#0 read in grayscale
equ = cv2.equalizeHist(img)
#clahe1=cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clahe1=cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(img)
clahe2=cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8)).apply(img)
clahe3=cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8,8)).apply(img)

clahe4=cv2.createCLAHE(clipLimit=2.0, tileGridSize=(32,32)).apply(img)
clahe5=cv2.createCLAHE(clipLimit=4.0, tileGridSize=(32,32)).apply(img)
clahe6=cv2.createCLAHE(clipLimit=10.0, tileGridSize=(32,32)).apply(img)
#bigger tileGridSize, big contrast in smaller areas
#bigger clipLimit, bigger contrst in the same areas

#equ_adaptive=clahe1.apply(img)
hist_img = cv2.calcHist([img],[0],None,[256],[0,256])
hist_equ = cv2.calcHist([equ],[0],None,[256],[0,256])
hist_clahe1 = cv2.calcHist([clahe1],[0],None,[256],[0,256])
hist_clahe2 = cv2.calcHist([clahe2],[0],None,[256],[0,256])
hist_clahe3 = cv2.calcHist([clahe3],[0],None,[256],[0,256])
hist_clahe4 = cv2.calcHist([clahe4],[0],None,[256],[0,256])
hist_clahe5 = cv2.calcHist([clahe5],[0],None,[256],[0,256])
hist_clahe6 = cv2.calcHist([clahe6],[0],None,[256],[0,256])

#try mask.. define area# for transform and or for histogram
plt.figure(figsize=(48,30))
plt.subplot(338)
plt.imshow(img, 'gray')
plt.title('original')

plt.subplot(337)
plt.imshow(equ, 'gray')
plt.title('equalized')

plt.subplot(331)
plt.imshow(clahe1, 'gray')
plt.title('adaptive_equalized1')

plt.subplot(332)
plt.imshow(clahe2, 'gray')
plt.title('adaptive_equalized2')

plt.subplot(333)
plt.imshow(clahe3, 'gray')
plt.title('adaptive_equalized3')

plt.subplot(334)
plt.imshow(clahe4, 'gray')
plt.title('adaptive_equalized4')

plt.subplot(335)
plt.imshow(clahe5, 'gray')
plt.title('adaptive_equalized5')

plt.subplot(336)
plt.imshow(clahe6, 'gray')
plt.title('adaptive_equalized6')

plt.subplot(339)
plt.plot(hist_img,label='original')
plt.plot(hist_equ,label='equalized')
plt.plot(hist_clahe1,label='adaptive_equalized1')
plt.plot(hist_clahe2,label='adaptive_equalized2')
plt.plot(hist_clahe3,label='adaptive_equalized3')
plt.plot(hist_clahe4,label='adaptive_equalized4')
plt.plot(hist_clahe5,label='adaptive_equalized5')
plt.plot(hist_clahe6,label='adaptive_equalized6')
plt.xlim([0,256])
plt.legend()
plt.title('histogram')

plt.savefig('../results/'+img_name+'_histEq.png')


#histograms
#https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_table_of_contents_histograms/py_table_of_contents_histograms.html#table-of-content-histograms

#template matching
#https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html#template-matching

#canny edge detection
#https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html#canny
