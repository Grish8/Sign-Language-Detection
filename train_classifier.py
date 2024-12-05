import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the data
data_dict = pickle.load(open('data.pickle', 'rb'))

# Check the maximum length of sequences
max_length = max(len(entry) for entry in data_dict['data'])

# Pad or truncate the sequences to the max length
data = np.array([np.pad(entry, (0, max_length - len(entry)), mode='constant') if len(entry) < max_length else entry[:max_length] for entry in data_dict['data']])

# Convert labels to a NumPy array
labels = np.asarray(data_dict['labels'])

# Ensure that labels match the length of data
assert len(data) == len(labels), "Mismatch between data and labels length."

# Split the data
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train the model
model = RandomForestClassifier()

model.fit(x_train, y_train)

# Predict and evaluate
y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly!'.format(score * 100))

# Save the model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
