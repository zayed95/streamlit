import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
import joblib

st.title('Diabetese Prediction Site')

st.info('This is an app that predicts whether you have diabetes or not by asking you a few questions about your overall mental and physical health')
df = pd.read_csv('https://raw.githubusercontent.com/zayed95/Diabetes/refs/heads/main/diabetes.csv')
with st.sidebar:
  st.header('Input Features')
  HighBP = st.toggle('High Blood Pressure')
  BMI = st.slider('BMI', 16.5, 90.0, 27.0)
  GenHlth = st.radio('Rate your general health with 1 being the best and 5 being the poorest', [1, 2, 3, 4, 5])
  PhysHlth = st.slider('How many days in the past month have you felt that your physical health was bad?', 0, 30, 0)
  MentHlth = st.slider('How many days in the past month have you felt that your mental health was bad?', 0, 30, 0)
  Age = st.slider('Age', 18, 80, 60)
  
# create a data frame for the input features
data = {
  'HighBP': HighBP,
  'BMI': BMI,
  'GenHlth': GenHlth,
  'PhysHlth': PhysHlth,
  'MentHlth': MentHlth,
  'Age': Age
}
input = pd.DataFrame(data, index=[0])
"""X = df[['HighBP', 'BMI', 'GenHlth',  'PhysHlth', 'MentHlth', 'Age']]
y = df[['Diabetes_binary']]
model = SVC(C=100, class_weight='balanced')
model.fit(X, y)"""
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()prediction = model.predict(input)

"""st.subheader('Diabetic')
if prediction:
  st.write('Yes')
else:
  st.write('No')"""
  
