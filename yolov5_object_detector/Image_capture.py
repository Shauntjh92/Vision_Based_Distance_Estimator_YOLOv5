import cv2
import numpy as np
import glob
import os

frameSize = (877, 306)

print(os.getcwd())
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, 30, frameSize, isColor=True)

for filename in glob.glob('runs/detect/exp281/*.jpg'):
    print(filename)
    img = cv2.imread(filename)
    out.write(img)

out.release()