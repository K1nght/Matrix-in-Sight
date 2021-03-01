import cv2 #opencv读取的格式是BGR
import matplotlib.pyplot as plt

img=cv2.imread('data/demo/0.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
x, y ,c = img.shape

scale = 5

BiCubic_big = cv2.resize(img ,(int(img.shape[1]*scale),int(img.shape[0]*scale)),interpolation=cv2.INTER_CUBIC)
cv2.imwrite('../data/demo/0_bigger.jpg', BiCubic_big)
plt.imshow(BiCubic_big)
plt.show()
