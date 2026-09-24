# TESTING THE MODEL
# Import necessary libraries
import joblib
import pandas as pd

# Load the model
model = joblib.load("models/best_knn.pkl")

# Create a test dataframe
new_student = pd.DataFrame({
    "number_courses": [6],
    "time_study": [7.0]
})

# Predict new student
prediction = model.predict(new_student)
print(prediction)