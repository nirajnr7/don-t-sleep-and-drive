'''we will use OpenCV's haarcascade (face and eye cascade) to detect  eyes & face
 in the image.'''

#Import necessary libraries
import cv2 as cv
import numpy as np

#Load face cascade and eye cascade
face_cascade = cv.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
eye_cascade = cv.CascadeClassifier("haarcascades/haarcascade_eye.xml")

#Read image as img and convert it to grayscale and store in gray.
img = cv.imread('images/test.jpg')
grayimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#Detect all faces which is in grayimg.
#scaleFactor: Parameter specifying how much the image size is reduced at each image scale.
#minNeighbors: Parameter specifying how many neighbors each candidate rectangle should have to retain it
#the face x,y w,h list is stored in face list
faces = face_cascade.detectMultiScale(grayimg, 1.3, 5)

#Draw a rectangle over the face
for (x,y,w,h) in faces:
    #we draw rectange in Red colour on original image for the face
    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    #roi is region of interest with area having face in it.
    roi_grayimg = grayimg[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

    #Detect eyes in face
    #detection is always in grey image
    eyes = eye_cascade.detectMultiScale(roi_grayimg)

    #we draw rectange in green  colour on original image for the eyes
    for (ex,ey,ew,eh) in eyes:
        cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

#final image is displayed
cv.imshow('Image', img)
cv.waitKey(0)
cv.destroyAllWindows()
