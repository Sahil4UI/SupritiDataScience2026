import cv2
# pip install opencv-python
# OPENCV  - CV - COmputer Vision

#image read(file name)
# image read
img = cv2.imread('img_42.jpg')
# cv2.imshow('result',img)
# print(img)
# HAAR CASCADE CLASSIFIER ->FACIAL RECOGNITIOn: 
#converting the image to gray scale
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#img, (x,y), (w,h), (color), borderWidth
# RGB - RED GREEN BLUE -255
cv2.rectangle(img,(20,20),(100,100),(255,0,0),5)
cv2.imwrite('result.jpg',img)

