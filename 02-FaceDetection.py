import cv2

dataset = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread('img_42.jpg')
# detect the face while zooming in
faces = dataset.detectMultiScale(img,3)
#print(faces)
for x,y,w,h in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,255),5)
cv2.imwrite('result.jpg',img)
