import streamlit as st
import pandas as pd
import joblib
import os

MODEL_PATH = "matchmaker_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found.")
    st.stop()

model = joblib.load(MODEL_PATH)

st.set_page_config(page_title="MatchmakerAI", page_icon="❤️")
st.title("❤️ MatchmakerAI")

essay = st.text_area("Profile Essay", height=200)
age = st.slider("Age", 18, 70, 25)

if st.button("Predict"):
    if essay.strip() == "":
        st.warning("Please enter an essay.")
        st.stop()

    input_df = pd.DataFrame({
        "essays": [essay],
        "age": [age]
    })

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df).max()

    if prediction == "single":
        st.success(f"💙 Prediction: SINGLE ({probability:.2%})")
    else:
        st.error(f"❤️ Prediction: NOT SINGLE ({probability:.2%})")
