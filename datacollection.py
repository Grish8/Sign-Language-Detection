import cv2  # Importing the OpenCV library for image processing
from cvzone.HandTrackingModule import HandDetector  # Importing HandDetector from the cvzone library
import numpy as np  # Importing NumPy for array manipulation
import math  # Importing math library for mathematical functions
import time  # Importing time library for handling time-related tasks
import mediapipe as mp


cap = cv2.VideoCapture(0)  # Initiating the webcam capture
detector = HandDetector(maxHands=1)  # Creating a hand detector object to detect at most one hand
offset = 20  # Setting an offset value for cropping the image
imgSize = 500  # Defining the size of the output image
counter = 0  # Initializing a counter for saved images

folder = "Data\\hello"  # Defining the folder path to save the images

print("Press 's' to save an image.")
print("Press 'q' to quit.")

while True:  # Starting an infinite loop
    success, img = cap.read()  # Capturing the frame from the webcam
    hands, img = detector.findHands(img)  # Detecting hands in the captured frame
    imgCrop = None  # Initialize imgCrop to None to avoid undefined variable issues

    if hands:  # If a hand is detected
        hand = hands[0]  # Taking the first detected hand
        x, y, w, h = hand['bbox']  # Getting the bounding box coordinates of the detected hand

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255  # Creating a white image of size imgSize x imgSize

        # Cropping the detected hand from the frame with an offset
        try:
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            imgCropShape = imgCrop.shape  # Getting the shape of the cropped image

            aspectRatio = h / w  # Calculating the aspect ratio of the hand

            if aspectRatio > 1:  # If height is greater than width
                k = imgSize / h  # Scaling factor based on height
                wCal = math.ceil(k * w)  # Calculating the new width
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))  # Resizing the image based on the height
                imgResizeShape = imgResize.shape  # Getting the shape of the resized image
                wGap = math.ceil((imgSize - wCal) / 2)  # Calculating the gap to center the image horizontally
                imgWhite[:, wGap: wCal + wGap] = imgResize  # Placing the resized image on the white image

            else:  # If width is greater than or equal to height
                k = imgSize / w  # Scaling factor based on width
                hCal = math.ceil(k * h)  # Calculating the new height
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))  # Resizing the image based on the width
                imgResizeShape = imgResize.shape  # Getting the shape of the resized image
                hGap = math.ceil((imgSize - hCal) / 2)  # Calculating the gap to center the image vertically
                imgWhite[hGap: hCal + hGap, :] = imgResize  # Placing the resized image on the white image

            cv2.imshow('ImageCrop', imgCrop)  # Displaying the cropped image
            cv2.imshow('ImageWhite', imgWhite)  # Displaying the resized image on a white background

        except Exception as e:
            print(f"Error during cropping/resizing: {e}")

    # Handle imgCrop if no hand was detected or cropping failed
    if imgCrop is None or (hasattr(imgCrop, 'size') and imgCrop.size == 0):
        print("Error: imgCrop is empty or None.")

    cv2.imshow('Image', img)  # Displaying the original frame
    key = cv2.waitKey(1)  # Waiting for a key press

    if key == ord("s"):  # If the 's' key is pressed
        if imgCrop is not None and imgCrop.size > 0:
            counter += 1  # Increment the counter
            cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)  # Save the image
            print(f"Saved image {counter}")
        else:
            print("Cannot save an empty or undefined cropped image.")
    
    if key == ord("q"):  # If the 'q' key is pressed
        print("Quitting the application...")
        break  # Exit the loop

cap.release()  # Release the webcam
cv2.destroyAllWindows()  # Close all OpenCV windows
