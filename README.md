# DAY-9

House Rent Prediction - Day 9/50 ML Challenge
This project is a machine learning-based tool designed to estimate monthly house rents across major Indian metropolitan cities. It focuses on implementing a clean, modular backend using Scikit-Learn pipelines and an interactive frontend using Streamlit.

📌 Project Overview
As part of a 50-day Machine Learning challenge, this "Day 9" project demonstrates the transition from raw data exploration to a structured, interactive application. The core objective is to handle diverse data types (numerical and categorical) through automated pipelines to predict rental prices with high reliability.

🚀 Features
Automated Preprocessing: Utilizes ColumnTransformer to handle numerical scaling and categorical encoding (One-Hot Encoding) in a single step.

Robust Pipeline: Bundles preprocessing and the RandomForestRegressor to prevent data leakage and ensure consistency.

Interactive UI: A clean, sidebar-driven interface built with Streamlit for real-time price estimation.

Custom Styling: Enhanced UI with custom CSS for a modern look and feel.

🛠 Tech Stack
Language: Python

Data Manipulation: Pandas

Machine Learning: Scikit-Learn

Model Persistence: Pickle

Web Interface: Streamlit

📊 Dataset
The model is trained on the House Rent Prediction Dataset, which includes features such as:

BHK (Number of bedrooms)

Size (Square footage)

Area Type (Super Area, Carpet Area, etc.)

City (Mumbai, Delhi, Bangalore, etc.)

Furnishing Status

Tenant Preference

Number of Bathrooms

📁 Project Structure
Plaintext
├── House_Rent_Dataset.csv   # Raw dataset
├── train.py                 # Training script & pipeline creation
├── app.py                   # Streamlit web application
├── house_rent_model.pkl     # Saved model pipeline (generated after training)
└── README.md
⚙️ How to Run
1. Install Requirements
Ensure you have the necessary libraries installed:

Bash
pip install pandas scikit-learn streamlit
2. Train the Model
Run the training script to generate the model pipeline:

Bash
python train.py
3. Launch the Application
Run the Streamlit app to interact with the model:

Bash
streamlit run app.py
🧠 Model Details
The project utilizes a Random Forest Regressor, an ensemble learning method that builds multiple decision trees to provide a more accurate and stable prediction. By wrapping the model in a Scikit-Learn Pipeline, the code remains modular and scalable for future updates.

Challenge Status: Day 9/50 🚀
Author: Rutuja Yatin Angare / Parth Jitendra Angare
License: MIT
