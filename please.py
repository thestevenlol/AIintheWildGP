import cv2 as cv
img = cv.imread("frame3.jpg")

cv.imshow("Display window", img)
k = cv.waitKey(0)