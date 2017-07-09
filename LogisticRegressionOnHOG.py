import numpy as np
import cv2
import os, sys
from sklearn.svm import SVC, LinearSVC
from skimage.measure import structural_similarity as ssim
import matplotlib.pyplot as plt
from my_cross_val import my_cross_val

h0=np.zeros((1,64))
h1=np.zeros((1,64))
h2=np.zeros((1,64))
h3=np.zeros((1,64))
h4=np.zeros((1,64))


bin_n = 16 # Number of bins
def hog(img):
  gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
  gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
  mag, ang = cv2.cartToPolar(gx, gy)
  bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
  bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
  mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
  hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
  hist = np.hstack(hists)     # hist is a 64 bit vector
  return hist


path_url="E:/uminn_notes/ComputerVision/Project/c0_segmented_resized/"
listdir = os.listdir(path_url)
for file in listdir:
  img = cv2.imread(path_url + file)
  h=hog(img)
  h0=np.vstack((h0,h))
h0=np.delete(h0,0,0) 
y0=np.zeros((2489,1))

path_url="E:/uminn notes/Computer Vision/Project/c1_segmented_resized/"
listdir = os.listdir(path_url)
for file in listdir:
  img = cv2.imread(path_url + file)
  h=hog(img)
  h1=np.vstack((h1,h))
h1=np.delete(h1,0,0)
y1=np.ones((2267,1)) 

path_url="E:/uminn notes/Computer Vision/Project/c2_segmented_resized/"
listdir = os.listdir(path_url)
for file in listdir:
  img = cv2.imread(path_url + file)
  h=hog(img)
  h2=np.vstack((h2,h))
h2=np.delete(h2,0,0)
y2=2*np.ones((2317,1))
 
path_url="E:/uminn notes/Computer Vision/Project/c5_segmented_resized/"
listdir = os.listdir(path_url)
for file in listdir:
  img = cv2.imread(path_url + file)
  h=hog(img)
  h3=np.vstack((h3,h))
h3=np.delete(h3,0,0)
y3=3*np.ones((2312,1)) 

path_url="E:/uminn notes/Computer Vision/Project/c6_segmented_resized/"
listdir = os.listdir(path_url)
for file in listdir:
  img = cv2.imread(path_url + file)
  h=hog(img)
  h4=np.vstack((h4,h))
h4=np.delete(h4,0,0)
y4=4*np.ones((2325,1)) 

X=np.vstack((h0,h1,h2,h3,h4))
y=np.vstack((y0,y1,y2,y3,y4))


X_y=np.hstack((X,y))

X_y=np.random.permutation(X_y)
X=X_y[:,0:63]
y=X_y[:,64]

Error_rate_kaggle_lsvc, mean_kaggle_lsvc, std_dev_kaggle_lsvc = my_cross_val("LogisticRegression",X,y,5)