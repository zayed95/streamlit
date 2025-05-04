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
        # Test prediction to verify model works
        test_input = pd.DataFrame({
            'HighBP': [1],
            'BMI': [25.0],
            'GenHlth': [3],
            'MentHlth': [5],
            'PhysHlth': [5],
            'Age': [50]
        })
        test_pred = model.predict(test_input)
        st.session_state.model_test_pred = test_pred[0]  # Store for debugging
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

model = load_model()

# Debug model loading
if 'model_test_pred' in st.session_state:
    st.write(f"Model test prediction (should be non-constant): {st.session_state.model_test_pred}")

# Input features
st.header('Input Features')
HighBP = int(st.toggle('High Blood Pressure'))  # Convert boolean to 0/1
BMI = st.slider('**BMI**', 16.5, 90.0, 27.0)
GenHlth = st.radio('General health (1=best, 5=poorest)', [1, 2, 3, 4, 5])
PhysHlth = st.slider('Bad physical health days (past month)', 0, 30, 0)
MentHlth = st.slider('Bad mental health days (past month)', 0, 30, 0)
Age = st.slider('Age', 18, 80, 60)

if st.button("Predict"):
    input_df = pd.DataFrame({
        'HighBP': [HighBP],
        'BMI': [BMI],
        'GenHlth': [GenHlth],
        'MentHlth': [MentHlth],
        'PhysHlth': [PhysHlth],
        'Age': [Age]
    })
    
    # Debug: show input data
    st.write("Input features:", input_df)
    
    try:
        prediction = model.predict(input_df)[0]
        st.write("Raw prediction value:", prediction)  # Debug output
        
        if prediction == 1:
            st.warning("Prediction: Diabetic")
        else:
            st.success("Prediction: Not Diabetic")
            
        # If available, show probabilities
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(input_df)[0]
            st.write(f"Probability: {proba[1]:.2%} chance of diabetes")
            
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
