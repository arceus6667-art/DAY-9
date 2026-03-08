import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
    page_title="RentVision AI",
    page_icon="🏠",
    layout="wide"
)

with open('house_rent_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
    }
    .result-card {
        padding: 2rem;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

st.sidebar.header("📍 Property Parameters")

with st.sidebar:
    city = st.selectbox("City", ['Kolkata', 'Mumbai', 'Bangalore', 'Delhi', 'Chennai', 'Hyderabad'])
    area_type = st.selectbox("Area Type", ['Super Area', 'Carpet Area', 'Built Area'])
    furnishing = st.selectbox("Furnishing Status", ['Unfurnished', 'Semi-Furnished', 'Furnished'])
    tenant = st.selectbox("Tenant Preferred", ['Bachelors/Family', 'Bachelors', 'Family'])
    
    st.divider()
    
    bhk = st.slider("BHK", 1, 6, 2)
    bathroom = st.slider("Bathrooms", 1, 5, 2)
    size = st.number_input("Size (sq ft)", min_value=10, max_value=8000, value=1000, step=50)

st.title("🏠 RentVision AI")
st.subheader("Predicting Urban Housing Costs with Machine Learning")

col1, col2 = st.columns([2, 1])

with col1:
    st.info("Adjust the parameters in the sidebar to estimate the monthly rental value for properties across India's major metropolitan hubs.")
    
    with st.expander("Why these factors matter?"):
        st.write("Location (City) and Size are the primary drivers of rent, while furnishing status and tenant preferences impact the final market listing.")

with col2:
    if st.button("Calculate Estimated Rent"):
        input_data = pd.DataFrame([[bhk, size, area_type, city, furnishing, tenant, bathroom]], 
                                  columns=['BHK', 'Size', 'Area Type', 'City', 'Furnishing Status', 'Tenant Preferred', 'Bathroom'])
        
        prediction = model.predict(input_data)[0]
        
        st.balloons()
        st.markdown(f"""
            <div class="result-card">
                <h3>Estimated Monthly Rent</h3>
                <h1 style='color: #ff4b4b;'>₹{round(prediction, 2):,}</h1>
                <p>Based on current market trends</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.write("### ← Set details and click calculate")

st.divider()
st.caption("Data source: House Rent Dataset | Model: Random Forest Regressor")
