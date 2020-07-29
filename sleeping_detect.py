'''This script detects if a person is drowsy or not,using dlib and eye aspect ratio
calculations. Uses webcam video feed as input.'''

#Import libraries
import time
import dlib
import cv2
from scipy.spatial import distance
from imutils import face_utils
import numpy as np
import pygame


#Initialize Pygame and load music
pygame.mixer.init()
pygame.mixer.music.load('audio/z.ogg')

#Minimum threshold of eye aspect ratio below which alarm is raise
Eye_aspect_ratio = 0.3

#Minimum consecutive frames for which eye ratio is below threshold for alarm raised
Eye_aspect_ratio_consec_frmaes = 30

#Counts no. of consecutuve frames below threshold
COUNTER = 0

#Load face cascade to draw a rectangle around detected faces.
face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")

#This function calculates and return eye aspect ratio
#eyes would look something like that
#    1  2
# 0        3
#    5  4
def eye_aspect_ratio(eye):
    a = distance.euclidean(eye[1], eye[5])
    b = distance.euclidean(eye[2], eye[4])
    c = distance.euclidean(eye[0], eye[3])

    ratio = (a+b) / (2*c)
    return ratio

# uses dlib shape predictor file,Load face detector and predictor,
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

#Extract indexes of facial landmarks for the left and right eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

#Start  video capture
video_capture = cv2.VideoCapture(0)

#Give some time for camera to initialize(not required)
time.sleep(3)

while(True):
    #Read each frame and flip it, and convert to grayscale
    ret, frame = video_capture.read()
    frame = cv2.flip(frame,1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detect all the  facial points through detector function
    faces = detector(gray, 0)

    #Detect faces through haarcascade_frontalface_default.xml
    face_rectangle = face_cascade.detectMultiScale(gray, 1.3, 5)

    #Draw rectangle around  face detected
    for (x,y,w,h) in face_rectangle:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    #Detect facial points in the frame
    for face in faces:

        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        #Get array of coordinates of eyes of both left and right
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        #Calculate aspect ratio of both eyes
        leftEyeAspectRatio = eye_aspect_ratio(leftEye)
        rightEyeAspectRatio = eye_aspect_ratio(rightEye)

        eyeAspectRatio = (leftEyeAspectRatio + rightEyeAspectRatio) / 2

        #Use hull to remove convex contour discrepencies
        # draw eye shape around eyes
        lefthull = cv2.convexHull(leftEye)
        righthull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [lefthull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [righthull], -1, (0, 255, 0), 1)

        #Detect if eye aspect ratio is less than threshold count
        if(eyeAspectRatio < Eye_aspect_ratio):
            COUNTER += 1
            #If threshold frames leser than no. of frames,
            if COUNTER >= Eye_aspect_ratio_consec_frmaes:
                pygame.mixer.music.play(-1)
                cv2.putText(frame, "You are Drowsy", (150,200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)
        else:
            pygame.mixer.music.stop()
            COUNTER = 0

    #Show video feed
    cv2.imshow('Video', frame)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break

#release the video capture and destroyAllWindows
video_capture.release()
cv2.destroyAllWindows()
