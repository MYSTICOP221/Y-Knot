import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Y-Knot Fraud Detective", layout="wide")

st.title("Y-Knot Fraud Detective")
st.markdown("Untangling Fraudulent Transactions - XGBoost + Streamlit")

# Load model
try:
    model = joblib.load('models/fraud_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    st.sidebar.success("Model is ready")
except:
    st.sidebar.error("Please run python train_model.py first")
    st.stop()

page = st.sidebar.radio("Navigation", ["Home", "Predict Transactions", "Performance", "About"])

if page == "Home":
    st.subheader("Project Overview")
    data = pd.read_csv('data/creditcard.csv')
    total = len(data)
    fraud_count = data['Class'].sum()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", f"{total:,}")
    col2.metric("Fraud Cases", fraud_count)
    col3.metric("Fraud Rate", f"{(fraud_count/total)*100:.4f}%")
    
    st.image('images/class_distribution.png', caption="Fraud vs Non-Fraud Distribution")

if page == "Predict Transactions":
    st.subheader("Batch Fraud Detection")
    uploaded_file = st.file_uploader("Upload transactions CSV", type="csv")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview", df.head())
        
        if st.button("Run Fraud Detection"):
            features = df.drop('Class', axis=1) if 'Class' in df.columns else df
            scaled = scaler.transform(features)
            predictions = model.predict(scaled)
            probabilities = model.predict_proba(scaled)[:, 1]
            
            df['Fraud_Predicted'] = predictions
            df['Fraud_Probability'] = (probabilities * 100).round(2)
            
            st.success(f"Analyzed {len(df)} transactions")
            st.dataframe(df)
            
            csv = df.to_csv(index=False)
            st.download_button("Download Predictions", csv, "fraud_predictions.csv")

if page == "Performance":
    st.subheader("Model Performance")
    st.write("Model trained with XGBoost on highly imbalanced data.")
    st.image('images/class_distribution.png', caption="Class Distribution")

if page == "About":
    st.markdown("""
    ### Y-Knot Fraud Detective
    
    This is an end-to-end machine learning project for credit card fraud detection.
    
    Features:
    - Handles severe class imbalance
    - XGBoost classifier
    - Interactive Streamlit dashboard
    - Batch prediction with downloadable results
    
    Author : Midhun Krishna
    """)