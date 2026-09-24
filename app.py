# MODEL DEPLOYMENT
# Importing necessary libraries
import joblib
import streamlit as st
import pandas as pd
# Add app title
st.title("Student Marks Predictor")

# Creating the model
model = joblib.load("models/best_knn.pkl")

# Add first user input
number_courses = st.number_input(
    "Number of Courses",
    min_value=1,
    max_value= 8,
    step=1
)
time_study = st.number_input(
    "Time Studied",
    min_value=0.1,
    max_value= 8.0,
    step=0.1
)

# Turn Streamlit Inputs into data the model expects
input_data = pd.DataFrame({
    "number_courses": [number_courses],
    "time_study": [time_study]
})

# Add button to trigger the prediction
if st.button("Predict Marks"):
    prediction = model.predict(input_data)

    st.success(
        f"Predicted Marks: {prediction[0]:.2f}"
    )