import cv2
from random import randrange
# randrange picks a color upto a given range excluding that range like if randrange(256) => will give number between 0-255

# Load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

# Choose an image to detect faces by calling image reaad function from opencv
img = cv2.imread('PIC.jpg')

# Must convert the img to grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces and coordinates of rectangles [[x-axis y-axis width height]] eg: [[140 182 577 577]] where first 2 values are x and y distance from top and topLeft to which next 2 coordinates are added to form rectangle
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# Draws rectangle around the faces with [[140 182 577 577]] coordinates and BGR color (here green) and a thickness of 2
# cv2.rectangle(img, (104, 182), (104+577, 182+577), (0, 255, 0), 2)
# (x, y, w, h) = face_coordinates[0]

# for (x, y, w, h) in range (len(face_coordinates)):
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(128, 256),
                                            randrange(256), randrange(256)), 2)

print(face_coordinates)

# cv2 function to display the read image faces
cv2.imshow('Face Detector', img)

# This keeps the window paused until a key is pressed
cv2.waitKey()

print("Code Completed")
