import os
import cv2

# Directory to save images
DATA_DIR = 'data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 3
dataset_size = 100

# Open the video capture
camera_index = 0  # Start with the first camera
cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    print(f"Error: Unable to access camera at index {camera_index}.")
    exit()

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Collecting data for class {}'.format(j))

    # Wait for user to start or skip class
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from the camera.")
            break

        cv2.putText(frame, 'Ready? Press "Q" to start, "N" to skip!', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(25) & 0xFF
        
        # If "Q" is pressed, start collecting data for this class
        if key == ord('q'):
            break
        # If "N" is pressed, skip to the next class
        elif key == ord('n'):
            print(f"Skipping class {j}. Moving to next class.")
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from the camera.")
            break

        # Display the current frame
        cv2.imshow('frame', frame)

        # Wait for the user to press the key
        key = cv2.waitKey(25) & 0xFF

        # If "N" is pressed, skip to the next class
        if key == ord('n'):
            print(f"Skipping class {j}. Moving to next class.")
            break

        # Save the image
        file_path = os.path.join(class_dir, '{}.jpg'.format(counter))
        cv2.imwrite(file_path, frame)
        counter += 1

# Release the camera and destroy windows
cap.release()
cv2.destroyAllWindows()
