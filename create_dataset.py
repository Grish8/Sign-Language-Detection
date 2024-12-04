import os
import pickle
import mediapipe as mp
import cv2

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize the hands module with a lower detection confidence
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.2)

DATA_DIR = 'data'

# Initialize data and labels
data = []
labels = []

# Check if DATA_DIR exists
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"The directory '{DATA_DIR}' does not exist. Please check the path.")

# Iterate through directories in DATA_DIR
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path):
        print(f"Skipping non-directory item: {dir_}")
        continue  # Skip files like .gitignore or other non-folder items

    print(f"Processing directory: {dir_}")
    for img_path in os.listdir(dir_path):
        img_file_path = os.path.join(dir_path, img_path)
        if not os.path.isfile(img_file_path):
            print(f"Skipping non-file item: {img_path}")
            continue

        try:
            data_aux = []
            x_ = []
            y_ = []

            # Load and process the image
            img = cv2.imread(img_file_path)
            if img is None:
                print(f"Error loading image: {img_file_path}")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Detect hand landmarks
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                # Process detected hands
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        x_.append(x)
                        y_.append(y)

                    # Normalize coordinates relative to the hand's position
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                data.append(data_aux)
                labels.append(dir_)
            else:
                print(f"No hands detected in image: {img_file_path}")

        except Exception as e:
            print(f"Error processing image {img_file_path}: {e}")

# Save the processed data
output_file = 'data.pickle'
with open(output_file, 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"Dataset successfully created and saved to '{output_file}'.")
