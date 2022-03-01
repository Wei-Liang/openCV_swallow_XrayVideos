import numpy as np
from img_processing_lib import *
video_name='mixed - Normal'
#mixed - Neuro
sec = 0
frameRate = 30
count=1

success = getFrame(video_name,sec,count)
while success:
    count = count + 1
    sec = sec + 1/frameRate
    sec = round(sec, 2)
    success = getFrame(video_name,sec,count)