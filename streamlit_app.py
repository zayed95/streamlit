import streamlit as st

st.title('Diabetese Prediction Site')

st.info('This is an app that predicts whether you have diabetes or not by asking you a few questions about your overall mental and physical health')

with st.sidebar:
  st.header('Input Features')
  HighBP = st.toggle('High Blood Pressure')
  BMI = st.slider('BMI', 16.5, 90.0, 27.0)
  GenHlth = st.radio('Rate your general health with 1 being the best and 5 being the poorest', [1, 2, 3, 4, 5])
  
