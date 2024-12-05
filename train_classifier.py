import pickle  # Import the pickle module for loading and saving serialized data

# Import necessary modules from scikit-learn for model creation, splitting datasets, and evaluation
from sklearn.ensemble import RandomForestClassifier  # RandomForestClassifier for classification tasks
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
from sklearn.metrics import accuracy_score  # For measuring the accuracy of the model
import numpy as np  # Import NumPy for efficient numerical computations

# Load the data dictionary from a serialized pickle file
data_dict = pickle.load(open('./data.pickle', 'rb'))  # Load the data and labels from 'data.pickle'

# Extract the data and labels arrays from the dictionary and convert them to NumPy arrays
data = np.asarray(data_dict['data'])  # Convert the feature data to a NumPy array
labels = np.asarray(data_dict['labels'])  # Convert the labels to a NumPy array

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    data,  # Feature data
    labels,  # Corresponding labels
    test_size=0.2,  # Allocate 20% of the data for testing
    shuffle=True,  # Shuffle the data before splitting
    stratify=labels  # Ensure class proportions are consistent in train and test sets
)

# Create an instance of the RandomForestClassifier
model = RandomForestClassifier()  # Initialize the Random Forest Classifier with default parameters

# Train the model using the training data
model.fit(x_train, y_train)  # Fit the model to the training features and labels

# Make predictions on the testing set
y_predict = model.predict(x_test)  # Predict the labels for the test data

# Calculate the accuracy of the model's predictions
score = accuracy_score(y_predict, y_test)  # Compare predicted labels with actual test labels to compute accuracy

# Print the accuracy score as a percentage
print('{}% of samples were classified correctly !'.format(score * 100))  # Display the classification accuracy

# Save the trained model to a pickle file
f = open('model.p', 'wb')  # Open a file named 'model.p' in write-binary mode
pickle.dump({'model': model}, f)  # Serialize and save the model in the file
f.close()  # Close the file after saving
