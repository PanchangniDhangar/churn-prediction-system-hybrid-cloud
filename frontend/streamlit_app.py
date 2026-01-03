import streamlit as st
import requests
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Telecom Churn Dashboard", layout="wide")

st.title("📡 Strategic Churn Prediction Dashboard")
st.markdown("Enter customer usage metrics to analyze churn risk.")

# --- 1. INITIALIZE USER INPUTS (The Fix) ---
# We define the defaults here so the dictionary exists before we use it in inputs
user_inputs = {
    "rev_Mean": 50.0,
    "totmrc_Mean": 45.0,
    "ovrmou_Mean": 0.0,
    "mou_Mean": 300.0,
    "months": 12,
    "hnd_price": 150.0,
    "phones": 1,
    "models": 1,
    "change_mou": 0.0,
    "change_rev": 0.0
}

# --- 2. CREATE DASHBOARD LAYOUT ---
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("💰 Billing")
    user_inputs["rev_Mean"] = st.number_input("Monthly Revenue ($)", value=user_inputs["rev_Mean"])
    user_inputs["totmrc_Mean"] = st.number_input("Monthly Charge ($)", value=user_inputs["totmrc_Mean"])
    user_inputs["ovrmou_Mean"] = st.number_input("Overage Minutes", value=user_inputs["ovrmou_Mean"])
    user_inputs["change_rev"] = st.slider("Revenue Change %", -100.0, 100.0, 0.0)

with col2:
    st.subheader("📱 Usage")
    user_inputs["mou_Mean"] = st.number_input("Avg Minutes (MOU)", value=user_inputs["mou_Mean"])
    user_inputs["change_mou"] = st.slider("MOU Change %", -100.0, 100.0, 0.0)
    user_inputs["months"] = st.number_input("Tenure (Months)", value=int(user_inputs["months"]), step=1)

with col3:
    st.subheader("📊 Device & Credit")
    user_inputs["phones"] = st.number_input("Number of Phones", value=int(user_inputs["phones"]), step=1)
    user_inputs["models"] = st.number_input("Number of Models", value=int(user_inputs["models"]), step=1)
    user_inputs["hnd_price"] = st.number_input("Handset Price ($)", value=user_inputs["hnd_price"])
    
    # Simple selection for categorical defaults
    refurb_new = st.selectbox("Refurbished Phone?", ["N", "Y"])
    creditcd = st.selectbox("Has Credit Card?", ["Y", "N"])
    
    # Update dictionary with selectbox values
    user_inputs["refurb_new"] = refurb_new
    user_inputs["creditcd"] = creditcd

st.divider()

# --- 3. PREDICTION LOGIC ---
if st.button("Analyze Risk", type="primary"):
    # Hybrid Cloud Config: Look for BACKEND_URL in secrets, default to local if not found
    backend_url = st.secrets.get("BACKEND_URL", "http://127.0.0.1:8000")
    predict_endpoint = f"{backend_url}/predict"
    
    with st.spinner('Connecting to FastAPI Backend...'):
        try:
            # Send the dictionary to the API
            response = requests.post(predict_endpoint, json={"data": user_inputs})
            
            if response.status_code == 200:
                result = response.json()
                prediction = result['prediction']
                
                if prediction == "Churn":
                    st.error(f"### High Risk: Customer likely to Churn")
                else:
                    st.success(f"### Low Risk: Customer likely to stay")
            else:
                st.warning(f"Backend returned an error: {response.status_code}")
                
        except Exception as e:
            st.error(f"Could not connect to Backend at {backend_url}. Make sure your FastAPI server is running.")