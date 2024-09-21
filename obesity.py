import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
import pickle

warnings.filterwarnings("ignore")

# Load the obesity data
data = pd.read_csv("dataset.csv")

# Extract features and target variable
X = data[['Age', 'Gender', 'Height', 'Weight', 'BMI', 'PhysicalActivityLevel']].values
y = data['ObesityCategory'].values

# Ensure the target variable is of integer type
y = y.astype('int')


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the logistic regression model
des = DecisionTreeClassifier()
des.fit(X_train, y_train)

# Predict on the test set and calculate accuracy
y_pred = des.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Example input (this should match the features in the dataset)
inputt = [34,0,145.3145708,87.11699097,41.25575499,2]  # Example values for Age, Gender, Height, Weight, BMI, physical_activity
final = [np.array(inputt)]

# Predict whether the input example is obese
is_obese = des.predict(final)
if is_obese == 0:
  result = "UnderWeight"
elif is_obese == 1:
  result = "Normal Weight"
elif is_obese == 2:
  result = "Overweight"
elif is_obese == 3:
 result = "Obese"

print(f"Is the person obese? {result}")

# Save the model using pickle
pickle.dump(des, open('model.pkl', 'wb'))

# Load the model to ensure it works correctly
model = pickle.load(open('model.pkl', 'rb'))
print(f"Model loaded: {model}")
