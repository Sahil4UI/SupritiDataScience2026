import cv2

dataset = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# switch on the camera
capture = cv2.VideoCapture(0)#camera on hojye

while True:
    ret, img = capture.read()#ye hume 2 cheejein dega - return,image
    if ret==True:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = dataset.detectMultiScale(gray,3)
        for x,y,w,h in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0),5)
        cv2.imshow('result',img)
        # 0XFF==27(Escape KEY)
        if cv2.waitKey(1) & 0xFF == 27:#0XFF->ESCAPE
            break
    else:
        print("Camera Not Working")

capture.release()
cv2.destroyAllWindows()

