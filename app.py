import streamlit as st
import joblib
import os

# IMPORTANT: Import ML classes BEFORE loading model
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="MatchmakerAI 💘",
    page_icon="💘",
    layout="centered"
)

st.title("💘 MatchmakerAI")
st.write("Predict whether a person is **Single** or **Not Single**")

# ----------------------------
# Load Model
# ----------------------------
MODEL_PATH = "model.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model file not found!")
        st.stop()
    return joblib.load(MODEL_PATH)

model = load_model()

# ----------------------------
# User Input
# ----------------------------
essay = st.text_area(
    "✍️ Describe yourself",
    placeholder="I enjoy traveling, meeting new people, and working on my career..."
)

age = st.number_input(
    "🎂 Age",
    min_value=18,
    max_value=100,
    value=25
)

# ----------------------------
# Prediction
# ----------------------------
if st.button("🔮 Predict Relationship Status"):
    if essay.strip() == "":
        st.warning("Please write a short description.")
    else:
        prediction = model.predict([[essay, age]])

        if prediction[0] == 1 or prediction[0] == "not_single":
            st.success("💑 Prediction: **Not Single**")
        else:
            st.info("💖 Prediction: **Single**")
