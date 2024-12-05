import os  # Import the os module for interacting with the operating system (e.g., file system operations)
import cv2  # Import OpenCV for image and video processing

# Directory where the dataset will be stored
DATA_DIR = 'data'

# Check if the data directory exists; if not, create it
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Define the number of classes (categories) for the dataset
number_of_classes = 3

# Define the number of images to collect for each class
dataset_size = 100

# Open a connection to the default camera (index 0)
cap = cv2.VideoCapture(0)

# Loop through each class to collect data
for j in range(number_of_classes):
    # Create a subdirectory for the current class if it doesn't exist
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    # Inform the user which class is being processed
    print('Collecting data for class {}'.format(j))

    done = False  # Flag to control readiness prompt
    while True:  # Display a readiness message to the user
        ret, frame = cap.read()  # Capture a frame from the camera
        # Add a message overlay on the frame to prompt the user
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)  # Show the frame to the user
        # Wait for the user to press 'q' to start data collection
        if cv2.waitKey(25) == ord('q'):
            break

    # Initialize a counter to track the number of images collected
    counter = 0
    while counter < dataset_size:  # Collect images until the dataset size is reached
        ret, frame = cap.read()  # Capture a frame from the camera
        cv2.imshow('frame', frame)  # Display the frame
        cv2.waitKey(25)  # Wait briefly to allow frame rendering
        # Save the current frame as an image file in the respective class directory
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)
        counter += 1  # Increment the counter

# Release the camera resource
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
