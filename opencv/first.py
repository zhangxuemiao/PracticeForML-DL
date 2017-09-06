import cv2
import matplotlib.pyplot as plt

img = cv2.imread('C:/Users/ZhangXuemiao/Pictures/zhangYan.jpg', cv2.IMREAD_UNCHANGED)
print(img.shape)
print(img[0, 0].shape)
print(img[0,0])

print(img.size)
print(img.dtype)
# plt.imshow(img)
# plt.show()
cv2.imshow('sss',img)