import pickle  # Import pickle for loading the saved model
import cv2  # Import OpenCV for video capture and image processing
import mediapipe as mp  # Import MediaPipe for hand detection and landmark estimation
import numpy as np  # Import NumPy for array manipulation

# Load the pre-trained model from a pickle file
model_dict = pickle.load(open('./model.p', 'rb'))  # Load the model dictionary from the file 'model.p'
model = model_dict['model']  # Extract the actual model object from the dictionary

# Initialize video capture from the default camera (device 0)
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands solution components
mp_hands = mp.solutions.hands  # MediaPipe Hands module for hand detection
mp_drawing = mp.solutions.drawing_utils  # Utility functions for drawing landmarks
mp_drawing_styles = mp.solutions.drawing_styles  # Predefined styles for landmarks and connections

# Create a MediaPipe Hands object with static image mode and minimum detection confidence
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define a dictionary to map model prediction outputs to labels
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3:'D', 4:'E'}  # Example labels for hand gesture classes

# Main loop for real-time video processing
while True:
    data_aux = []  # List to store normalized hand landmark data for the current frame
    x_ = []  # List to store x-coordinates of landmarks
    y_ = []  # List to store y-coordinates of landmarks

    # Capture a frame from the video feed
    ret, frame = cap.read()  # `ret` indicates if the frame was captured successfully, `frame` is the captured image
    if not ret:
        print("Failed to capture frame from camera. Exiting...")
        break

    H, W, _ = frame.shape  # Get the height, width, and channels of the frame

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert the frame from BGR to RGB (MediaPipe requires RGB)

    results = hands.process(frame_rgb)  # Process the frame to detect hand landmarks

    if results.multi_hand_landmarks:  # Check if hand landmarks were detected
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the detected hand landmarks on the frame
            mp_drawing.draw_landmarks(
                frame,  # The frame to draw on
                hand_landmarks,  # The detected hand landmarks
                mp_hands.HAND_CONNECTIONS,  # Draw connections between landmarks
                mp_drawing_styles.get_default_hand_landmarks_style(),  # Style for landmarks
                mp_drawing_styles.get_default_hand_connections_style())  # Style for connections

        # Process each detected hand
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x  # Extract the x-coordinate
                y = hand_landmarks.landmark[i].y  # Extract the y-coordinate

                x_.append(x)  # Append x-coordinate to the list
                y_.append(y)  # Append y-coordinate to the list

            # Normalize landmark coordinates relative to the minimum x and y values
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))  # Normalize x-coordinate
                data_aux.append(y - min(y_))  # Normalize y-coordinate

    # Ensure the input data matches the model's expected number of features (84)
    while len(data_aux) < 84:
        data_aux.append(0)  # Pad with zeros if features are missing

    # If no landmarks were detected, skip the prediction step
    if len(data_aux) == 0:
        continue

    try:
        # Predict the hand gesture using the model
        prediction = model.predict([np.asarray(data_aux)])  # Pass the normalized data to the model for prediction

        # Map the predicted class index to the corresponding label
        predicted_character = labels_dict[int(prediction[0])]

        # Display the predicted character above the rectangle
        cv2.putText(frame, predicted_character, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3,
                    cv2.LINE_AA)
    except Exception as e:
        print(f"Prediction error: {e}")

    # Display the processed frame with predictions
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit the loop if 'q' is pressed
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
