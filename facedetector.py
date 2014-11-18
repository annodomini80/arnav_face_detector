import numpy as np
import cv2
import sys
from scipy import ndimage
import random

face_cascade = cv2.CascadeClassifier('../OpenCV/opencv-2.4.10/data/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def rotateImage(image, angle):
	col,row,n = image.shape
	print n,row,col
	center=tuple(np.array([col,row])/2)
	print center
    	rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
   	new_image = cv2.warpAffine(image, rot_mat, (col,row))
    	return new_image

def cropImage(img, x, y, w, h):
	crop_img = img[y:y+h, x:x+w]
	return crop_img

count = 1

if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        for line in f:
            img = cv2.imread(line.strip())
            if img is None:
                continue
            
            img_resize=cv2.resize(img, (0,0), fx=0.25, fy=0.25)
            img_rotate = ndimage.rotate(img_resize, -90)
            gray = cv2.cvtColor(img_rotate, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=10)#minSize=(30, 30),#flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
            
            for (x,y,w,h) in faces:
                
                if faces is not None:
                    crop_fname = "pos_data/" + str(random.getrandbits(256)) + ".png"
                    cv2.imwrite(crop_fname, cropImage(img_rotate,x,y,w,h))
                cv2.rectangle(img_rotate,(x,y),(x+w,y+h),(256,0,0),5)
                    
                fname = "res/" + str(count) + ".png"
                cv2.imwrite(fname, img_rotate)
                count=count+1
                print fname
                cv2.imshow('img', img_rotate);
                cv2.waitKey(10)
