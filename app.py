{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "774f3eed-539e-42e7-85e3-04afb21b405e",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import joblib\n",
    "import numpy as np\n",
    "\n",
    "# Load the trained model\n",
    "model = joblib.load('iris_model.pkl')\n",
    "\n",
    "# Load the scaler\n",
    "scaler = joblib.load('scaler.pkl')\n",
    "\n",
    "# Load the label encoder\n",
    "label_encoder = joblib.load('label_encoder.pkl')\n",
    "class_names = label_encoder.classes_\n",
    "\n",
    "st.title('Iris Flower Prediction App')\n",
    "st.write('Enter the sepal and petal measurements to predict the Iris species.')\n",
    "\n",
    "sepal_length = st.slider('Sepal Length (cm)', 4.0, 8.0, 5.0)\n",
    "sepal_width = st.slider('Sepal Width (cm)', 2.0, 4.5, 3.0)\n",
    "petal_length = st.slider('Petal Length (cm)', 1.0, 7.0, 4.0)\n",
    "petal_width = st.slider('Petal Width (cm)', 0.1, 2.5, 1.3)\n",
    "\n",
    "if st.button('Predict'):\n",
    "    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])\n",
    "\n",
    "    # Scale the input data\n",
    "    scaled_data = scaler.transform(input_data)\n",
    "\n",
    "    prediction_encoded = model.predict(scaled_data)[0]\n",
    "    predicted_class = class_names[prediction_encoded]\n",
    "\n",
    "    st.write(f'## Prediction: {predicted_class}')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
