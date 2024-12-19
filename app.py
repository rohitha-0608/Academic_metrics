import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os
import base64

# Loading the trained model
with open('student_mark_predictor.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to read and encode the image file
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string

# Set the background image using CSS
def set_background(image_base64):
    page_bg_img = f"""
    <style>
    .stApp {{
        background: url("data:image/png;base64,{image_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        color: white;
    }}
    .css-1g8v9l0 {{
        background: rgba(255, 255, 255, 0.8);
        padding: 20px;
        border-radius: 10px;
        text-align: center; /* Centering the text */
    }}
    h1, h2, h3, h4, h5, h6, p, span, div, label {{
        color: white;
    }}
    .stButton > button {{
        background-color: #4C4C6D;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
    }}
    .stButton > button:hover {{
        background-color: #6A5ACD;
    }}
    .stSlider > div {{
        background-color: transparent;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Call the function with the uploaded background image
image_base64 = get_base64_image("image3.jpg")
set_background(image_base64)

# Function to predict marks
def predict_marks(study_hours):
    return model.predict([[study_hours]])[0][0]

# Function to save input and prediction to a CSV file
def save_to_csv(study_hours, predicted_marks):
    # Check if the CSV file exists, if not create it with headers
    file_path = 'predictions.csv'
    if not os.path.isfile(file_path):
        df = pd.DataFrame(columns=["Study Hours", "Predicted Marks"])
        df.to_csv(file_path, index=False)

    # Append the new data to the CSV file
    new_data = pd.DataFrame({"Study Hours": [study_hours], "Predicted Marks": [predicted_marks]})
    new_data.to_csv(file_path, mode='a', header=False, index=False)

# Dashboard title
# Dashboard title with emoji
st.markdown("<h1 style='text-align: center;'>Academic Performance Analyzer</h1>", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)  # Horizontal line
st.markdown("<br>", unsafe_allow_html=True)

# Input for study hours with minimum of 1 hour and maximum of 23 hours
study_hours = st.number_input("Enter Study Hours:", min_value=1.0, max_value=23.0, value=1.0)

# Function to predict marks
def predict_marks(study_hours):
    # Predict marks using the model
    predicted_marks = model.predict([[study_hours]])[0][0]
    
    # Clip the predicted marks to be between 0 and 100
    return max(0, min(predicted_marks, 99))

# Button to trigger prediction
if st.button("Predict"):
    predicted_marks = predict_marks(study_hours)
    # Display predicted marks in a larger, bold font
    st.markdown(f"<h3 style='font-weight: bold;'>Predicted Marks for {study_hours} hours of study: <span style='color: blue;'>{predicted_marks:.2f} %</span></h2>", unsafe_allow_html=True)

    # Add a gap and a horizontal line
    st.markdown("<hr>", unsafe_allow_html=True)  # Horizontal line
    st.markdown("<br>", unsafe_allow_html=True)  # Gap

    # Save input and prediction to CSV
    save_to_csv(study_hours, predicted_marks)

    # Display model performance metrics
    mse = 1.108  # Example value, replace with actual MSE
    r2 = 0.951  # Example value, replace with actual R²
    mae = 0.8780690208883186  # Replace with actual MAE computation

    # Show performance metrics
    st.subheader("Model Performance Metrics:")
    st.write(f"Mean Squared Error (MSE): **{mse:.3f}**")
    st.write(f"R-squared (R²): **{r2:.3f}**")
    st.write(f"Mean Absolute Error (MAE): **{mae:.3f}**")

# Footer
st.markdown("---")
st.markdown("### About this Project")
st.write("This dashboard predicts student marks based on study hours using a linear regression model. "
         "The model's performance is evaluated using various metrics.")
