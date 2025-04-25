import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('iris_model.pkl')

# Load the scaler
scaler = joblib.load('scaler.pkl')

# Load the label encoder
label_encoder = joblib.load('label_encoder.pkl')
class_names = label_encoder.classes_

st.title('Iris Flower Prediction App')
st.write('Enter the sepal and petal measurements to predict the Iris species.')

sepal_length = st.slider('Sepal Length (cm)', 4.0, 8.0, 5.0)
sepal_width = st.slider('Sepal Width (cm)', 2.0, 4.5, 3.0)
petal_length = st.slider('Petal Length (cm)', 1.0, 7.0, 4.0)
petal_width = st.slider('Petal Width (cm)', 0.1, 2.5, 1.3)

if st.button('Predict'):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Scale the input data
    scaled_data = scaler.transform(input_data)

    prediction_encoded = model.predict(scaled_data)[0]
    predicted_class = class_names[prediction_encoded]

    st.write(f'## Prediction: {predicted_class}')