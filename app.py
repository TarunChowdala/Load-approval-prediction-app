import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf

# Load model and preprocessors
model = tf.keras.models.load_model('model.h5')

with open('label_encoder_married.pkl', 'rb') as file:
    label_encoder_married = pickle.load(file)

with open('one_hot_encoded_EmpStatus.pkl', 'rb') as file:
    one_hot_encoded_EmpStatus = pickle.load(file)

with open('Scaler.pkl', 'rb') as file:
    Scaler = pickle.load(file)

st.title("Loan Approval Prediction App")
st.write("Enter customer details to predict loan approval")

# Input fields based on test_data.csv structure
col1, col2 = st.columns(2)

with col1:
    age = st.slider('Age', 21, 59, 35)
    income = st.slider('Income', 20000, 150000, 50000)
    credit_score = st.slider('Credit Score', 300, 849, 600)
    loan_amount = st.slider('Loan Amount', 50000, 500000, 200000)

with col2:
    existing_loans = st.slider('Existing Loans', 0, 3, 0)
    married = st.selectbox('Married', label_encoder_married.classes_)
    employment_status = st.selectbox('Employment Status', one_hot_encoded_EmpStatus.categories_[0])

# Create input dataframe
input_data = {
    'Age': age,
    'Income': income,
    'CreditScore': credit_score,
    'LoanAmount': loan_amount,
    'Married': married,
    'ExistingLoans': existing_loans,
    'EmploymentStatus': employment_status
}

input_df = pd.DataFrame([input_data])

# Preprocess: Encode Married
input_df['Married'] = label_encoder_married.transform(input_df['Married'])

# Preprocess: One-hot encode EmploymentStatus
emp_status_encoded = one_hot_encoded_EmpStatus.transform(input_df[['EmploymentStatus']]).toarray()
emp_status_df = pd.DataFrame(
    emp_status_encoded,
    columns=one_hot_encoded_EmpStatus.get_feature_names_out(['EmploymentStatus'])
)

# Combine: Drop EmploymentStatus and add encoded columns
input_df = pd.concat([input_df.drop('EmploymentStatus', axis=1), emp_status_df], axis=1)

# Scale the features
input_scaled = Scaler.transform(input_df)

# Predict
if st.button('Predict Loan Approval'):
    prediction = model.predict(input_scaled)
    prediction_proba = prediction[0][0]
    
    st.subheader("Prediction Result")
    
    if prediction_proba > 0.5:
        st.success(f"✅ Loan Approved!")
        st.write(f"Approval Probability: {prediction_proba*100:.2f}%")
    else:
        st.error(f"❌ Loan Not Approved")
        st.write(f"Approval Probability: {prediction_proba*100:.2f}%")
    
    # Show probability bar
    st.progress(float(prediction_proba))
    st.caption(f"Confidence: {prediction_proba*100:.1f}%")