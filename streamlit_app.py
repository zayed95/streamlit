import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
import joblib

st.title('Diabetese Prediction Site')

st.info('This is an app that predicts whether you have diabetes or not by asking you a few questions about your overall mental and physical health')
df = pd.read_csv('https://raw.githubusercontent.com/zayed95/Diabetes/refs/heads/main/diabetes.csv')
st.header('Input Features')
HighBP = int(st.toggle('High Blood Pressure'))
st.divider()
BMI = st.slider('**BMI**', 16.5, 90.0, 27.0)
st.divider()
GenHlth = st.radio('Rate your general health with 1 being the best and 5 being the poorest', [1, 2, 3, 4, 5])
st.divider()
PhysHlth = st.slider('How many days in the past month have you felt that your **physical health** was bad?', 0, 30, 0)
st.divider()
MentHlth = st.slider('How many days in the past month have you felt that your **mental health** was bad?', 0, 30, 0)
st.divider()
Age = st.slider('Age', 18, 80, 60)
st.divider()

# create a data frame for the input features
data = {
  'HighBP': HighBP,
  'BMI': BMI,
  'GenHlth': GenHlth,
  'MentHlth': MentHlth,
  'PhysHlth': PhysHlth,
  'Age': Age
}

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

if st.button("Predict"):
    input = pd.DataFrame(data, index=[0])
    prediction = model.predict(input)[0]
    if prediction == 1:
        st.warning(f"Prediction: Diabetic")
    else:
        st.success(f"Prediction: Not Diabetic")
