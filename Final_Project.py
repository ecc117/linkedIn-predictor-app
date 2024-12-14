#!/usr/bin/env python
# coding: utf-8

# Final Project: LinkedIn User Prediction App
# Elias Carson - 12/06/2024

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Load dataset
s = pd.read_csv("social_media_usage.csv")

# Data Cleaning Function
def clean_sm(x):
    return np.where(x == 1, 1, 0)

# Prepare the dataset
ss = pd.DataFrame({
    'sm_li': s['web1h'].apply(clean_sm),
    'income': np.where(s['income'] > 9, np.nan, s['income']),
    'education': np.where(s['educ2'] > 8, np.nan, s['educ2']),
    'parent': s['par'].apply(clean_sm),
    'married': s['marital'].apply(clean_sm),
    'female': np.where(s['gender'] == 2, 1, 0),
    'age': np.where(s['age'] > 98, np.nan, s['age'])
}).dropna()

# Set up features and target
x = ss[['income', 'education', 'parent', 'married', 'female', 'age']]
y = ss['sm_li']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=202)

# Train logistic regression model
lr = LogisticRegression(class_weight='balanced', random_state=202)
lr.fit(x_train, y_train)

# App Title
st.title("LinkedIn User Prediction App")
st.write("This app predicts whether a person is a LinkedIn user based on their demographic information.")

# User Input Form
with st.form("user_input_form"):
    income = st.slider("Income (1-9)", min_value=1, max_value=9, value=5, help="1: Lowest income, 9: Highest income")
    education = st.slider("Education Level (1-8)", min_value=1, max_value=8, value=4, help="1: Less than high school, 8: Advanced degree")
    parent = st.selectbox("Are you a parent?", options=["No", "Yes"])
    married = st.selectbox("Are you married?", options=["No", "Yes"])
    female = st.selectbox("Are you female?", options=["No", "Yes"])
    age = st.slider("Age", min_value=18, max_value=98, value=30)
    submit = st.form_submit_button("Predict")

# User Input and Predictions
if submit:
    # Create a DataFrame for user input
    user_data = pd.DataFrame({
        'income': [income],
        'education': [education],
        'parent': [1 if parent == "Yes" else 0],
        'married': [1 if married == "Yes" else 0],
        'female': [1 if female == "Yes" else 0],
        'age': [age]
    })

    # Predict target class and probability
    prediction = lr.predict(user_data)[0]
    probability = lr.predict_proba(user_data)[0][1]

    # Display Results
    st.subheader("Prediction Results")
    st.write(f"**Prediction:** {'LinkedIn User' if prediction == 1 else 'Not a LinkedIn User'}")
    st.write(f"**Probability of being a LinkedIn User:** {probability:.2%}")



# Ensure newdata only contains the original feature columns
newdata_cleaned = newdata[['income', 'education', 'parent', 'married', 'female', 'age']]

# Predict target class
newdata['linkedin_user_prediction'] = lr.predict(newdata_cleaned)

# Predict probabilities for each class
newdata['prediction_probability'] = lr.predict_proba(newdata_cleaned)[:, 1]

# Display the updated DataFrame with predictions
st.dataframe(newdata)

