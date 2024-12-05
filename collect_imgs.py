import os  # For interacting with the file system
import cv2  # For image and video processing

# Directory where the dataset will be stored
DATA_DIR = 'data'

# Check if the data directory exists; if not, create it
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Open a connection to the default camera (index 0)
cap = cv2.VideoCapture(0)

# Inform the user about the process
print("Press 'q' to quit or 'n' to skip to the next class during data collection.")

while True:
    # Prompt the user to enter the class name
    class_name = input("Enter the class name (or type 'exit' to stop): ").strip()

    if class_name.lower() == 'exit':  # Stop if the user types 'exit'
        print("Exiting the program.")
        break

    # Create a subdirectory for the current class if it doesn't exist
    class_path = os.path.join(DATA_DIR, class_name)
    if not os.path.exists(class_path):
        os.makedirs(class_path)

    print(f'Collecting data for class "{class_name}"')

    # Inform the user to get ready
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, f'Ready? Press "Q" to start for {class_name}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 
                    (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(25)

        if key == ord('q'):  # Start collecting data
            break
        elif key == ord('n'):  # Skip this class
            print(f'Skipping class "{class_name}".')
            break

    # If the user chose to skip, go to the next iteration
    if key == ord('n'):
        continue

    # Initialize a counter for collected images
    counter = 0
    dataset_size = 100  # Number of images per class
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)

        key = cv2.waitKey(25)
        if key == ord('n'):  # Allow the user to skip this class
            print(f'Skipping class "{class_name}" during collection.')
            break

        # Save the current frame as an image in the respective class directory
        file_path = os.path.join(class_path, f'{counter}.jpg')
        cv2.imwrite(file_path, frame)
        counter += 1

    print(f'Data collection for class "{class_name}" completed.')

# Release the camera resource
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
