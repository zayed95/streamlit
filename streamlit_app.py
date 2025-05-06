import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title('Diabetes Prediction Site')

# Load model with verification
@st.cache_resource
def load_model():
    try:
        model = joblib.load("model.pkl")
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

model = load_model()

# Input features
st.header('Input Features')
st.divider()
HighBP = st.radio('**High Blood Pressure**', ['Yes', 'No'])
BMI = st.slider('**BMI**', 16.5, 90.0, 27.0)
GenHlth = st.radio('**General health** (1=best, 5=poorest)', [1, 2, 3, 4, 5])
PhysHlth = st.slider('**Bad physical health days** (past month)', 0, 30, 0)
MentHlth = st.slider('**Bad mental health days** (past month)', 0, 30, 0)
Age = st.slider('**Age**', 18, 80, 60)

if st.button("Predict"):
    if HighBP == 'Yes':
        HighBP = 1
    else: 
        HighBP = 0
    input_df = pd.DataFrame({
        'HighBP': [HighBP],
        'BMI': [BMI],
        'GenHlth': [GenHlth],
        'MentHlth': [MentHlth],
        'PhysHlth': [PhysHlth],
        'Age': [Age]
    })
    # Debug: show input data
    st.write("Input features are:", input_df)
    
    try:
        prediction = model.predict(input_df)[0]
        st.write("Raw prediction value:", prediction)
        
        if prediction:
            st.warning("Prediction: Diabetic")
        else:
            st.success("Prediction: Not Diabetic")
            
        # If available, show probabilities
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(input_df)[0]
            st.write(f"Probability: {proba[1]:.2%} chance of diabetes")
            
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")


