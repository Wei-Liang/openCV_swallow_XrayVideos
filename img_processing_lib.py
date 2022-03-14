import cv2
from matplotlib import pyplot as plt
import numpy as np
def getFrame(video_name,sec,count):
    vidcap = cv2.VideoCapture('../data/'+video_name+'.mp4')
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        cv2.imwrite('../data/'+video_name+'__'+str(count)+'.jpg', image)     # save frame as JPG file
    return hasFrames


def compareHistogramEqualizers(img_name):
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
    plt.close()


def equalilzeHistogram(img_name,method,clipLimit=4.0, tileGridSize=(32,32)):
    img=cv2.imread('../data/'+img_name+'.jpg',0)
    if method=='equalizeHist':
        equ = cv2.equalizeHist(img)
    elif method == 'CLAHE':
        equ=cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize).apply(img)
    else:
        print('equalizer not implemented')
    cv2.imwrite('../data/'+img_name+'_equ.jpg', equ)


def siftAndHomo(queryImg_name,img_name):
    img1 = cv2.imread('../data/'+queryImg_name+'.jpg',0) 
    img2 = cv2.imread('../data/'+img_name+'.jpg',0) # trainImage
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    MIN_MATCH_COUNT = 10

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)#see what are detected?
    kp2, des2 = sift.detectAndCompute(img2,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    else:
        print('Not enough matches are found - %d/%d' % (len(good),MIN_MATCH_COUNT))
        matchesMask = None


    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)


    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

    plt.figure()
    plt.imshow(img3, 'gray')
    plt.savefig('../results/'+img_name+'_'+queryImg_name+'_Detection.png', dpi=300)
    plt.close()
