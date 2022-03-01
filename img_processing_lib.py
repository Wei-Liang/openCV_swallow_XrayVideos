import cv2
def getFrame(video_name,sec,count):
    vidcap = cv2.VideoCapture('../data/'+video_name+'.mp4')
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        cv2.imwrite('../data/'+video_name+'__'+str(count)+".jpg", image)     # save frame as JPG file
    return hasFrames