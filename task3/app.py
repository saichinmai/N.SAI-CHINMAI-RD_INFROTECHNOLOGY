import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("disease_model.pkl", "rb"))

st.set_page_config(
    page_title="Medical Diagnosis System",
    layout="wide"
)

st.title("AI-Powered Medical Diagnosis System")

st.write("Select symptoms to predict disease")

# Symptoms
fever = st.checkbox("Fever")
cough = st.checkbox("Cough")
headache = st.checkbox("Headache")
vomiting = st.checkbox("Vomiting")
fatigue = st.checkbox("Fatigue")
body_pain = st.checkbox("Body Pain")
cold = st.checkbox("Cold")
chest_pain = st.checkbox("Chest Pain")
breathing_problem = st.checkbox("Breathing Problem")

# Predict button
if st.button("Predict Disease"):

    symptoms = np.array([[
        int(fever),
        int(cough),
        int(headache),
        int(vomiting),
        int(fatigue),
        int(body_pain),
        int(cold),
        int(chest_pain),
        int(breathing_problem)
    ]])

    prediction = model.predict(symptoms)[0]

    st.success(f"Predicted Disease: {prediction}")