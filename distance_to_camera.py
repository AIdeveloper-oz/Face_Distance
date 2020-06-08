# USAGE
# python distance_to_camera.py

# import the necessary packages
from imutils import paths
import numpy as np
import imutils
import cv2
import sys
import time

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

def find_marker(image):
	# convert the image to grayscale, blur it, and detect edges
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(gray, 35, 125)
	
	# find the contours in the edged image and keep the largest one;
	# we'll assume that this is our piece of paper in the image
	cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	c = max(cnts, key = cv2.contourArea)

	# compute the bounding box of the of the paper region and return it
	print(cv2.minAreaRect(c))
	return cv2.minAreaRect(c)

def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
	return (knownWidth * focalLength) / perWidth

# initialize the known distance from the camera to the object, which
# in this case is 24 inches
KNOWN_DISTANCE = 24.0

# initialize the known object width, which in this case, the piece of
# paper is 12 inches wide
KNOWN_WIDTH = 7.6

# load the furst image that contains an object that is KNOWN TO BE 2 feet
# from our camera, then find the paper marker in the image, and initialize
# the focal length
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
arr=np.zeros([100000])
image = cv2.imread("images/2ft.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags=cv2.CASCADE_SCALE_IMAGE
)
# Draw a rectangle around the faces
kj=0
for (x, y, w, h) in faces:
	cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
	arr[kj]=w
	kj=kj+1
	print(arr[kj-1])
	marker=arr[kj-1]
	focalLength = (marker * KNOWN_DISTANCE) / KNOWN_WIDTH
	print(focalLength)



video_capture = cv2.VideoCapture(0)




# loop over the images
for imagePath in sorted(paths.list_images("images")):
	# load the image, find the marker in the image, then compute the
	# distance to the marker from the camera
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(
   		gray,
    	scaleFactor=1.1,
    	minNeighbors=5,
    	minSize=(30, 30),
    	flags=cv2.CASCADE_SCALE_IMAGE
	)
	# Draw a rectangle around the faces
	kj=0
	for (x, y, w, h) in faces:
		cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
		arr[kj]=w
		kj=kj+1
		print(arr[kj-1])
		marker=arr[kj-1]
		focalLength = (marker * KNOWN_DISTANCE) / KNOWN_WIDTH
		print(focalLength)
		marker = arr[kj-1]
		inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker)
	
	# draw a bounding box around the image and display it
	
	cv2.putText(image, "%.2fft" % (inches / 12),(image.shape[1] - 200, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,2.0, (0, 255, 0), 3)
	#cv2.imshow("image", image)
	print((inches / 12))
	cv2.waitKey(0)



while(True):
	ret, frame = video_capture.read()
	frame=cv2.flip(frame,1)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    # Draw a rectangle around the faces
	kj=0
	flag=0
	for (x, y, w, h) in faces:
		flag=1
		arr[kj]=w
		kj=kj+1
		print(arr[kj-1])
		marker=arr[kj-1]
		inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker)
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
		cv2.putText(frame, "%.2fft" % (inches / 12),(x,y-10), cv2.FONT_HERSHEY_SIMPLEX,1.0, (0, 255, 0), 2)
		cv2.imshow("image", frame)

    # Display the resulting frame
	 
	if flag==0:
		cv2.imshow("image",frame)
	
	time.sleep(0.25)
	k=cv2.waitKey(1)
	if k==27:
		break

video_capture.release()
cv2.destroyAllWindows()