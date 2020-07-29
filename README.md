 ## Requirements


##The following project is written in python 3.7 and used the library openCV.The project is using real time video which is captured by the camera. The main objective of this program is to analyse whether the person is sleeping.If so then raise the alarm.

=======
IMPORTANT

  Download `shape_predictor_68_face_landmarks.dat.bz2` from [Shape Predictor 68 features](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
  Extract the file in the project folder using
 HEAD

  library used are-
=======
  ``bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2``



  numpy==1.15.2
	dlib==19.16.0
	pygame==1.9.4
	imutils==0.5.1
	opencv_python==3.4.3.18
	scipy==1.1.0

Use `pip install -r requirements.txt`to install the given requirements.


## "making_rectangle_on_face_and_eyes_single_image.py" will help you to understand the concept of how we use CascadeClassifier to detect eye and face on a single image.

## "making_rectangle_on_face_and_eyes_webcam.py" will help you to understand the concept of how we use CascadeClassifier to detect eye and face on a live video feed.

## "sleeping_detect.py" is the main file of project.


