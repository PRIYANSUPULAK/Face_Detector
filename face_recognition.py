import cv2
import sys
image_path=sys.argv[1]
cascPath = "haarcascade_frontalface_default.xml"

faceCascade=cv2.CascadeClassifier(cascPath)

image=cv2.imread(image_path)
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

faces= faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=9, minSize=(30, 30), flags= cv2.CASCADE_SCALE_IMAGE)

for (x,y,w,h) in faces:
	cv2.rectangle(image, (x,y),(x+w,y+h),(0,255,0),2)

cv2.imshow("Faces found",image)
#cv2.waitkey(0)	
cv2.waitKey(0)
