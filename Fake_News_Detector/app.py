import streamlit as st
import pickle
import os

BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")

model = pickle.load(open(model_path, "rb"))
vectorizer = pickle.load(open(vectorizer_path, "rb"))

st.title("📰 Fake News Detector")

news = st.text_area("Enter News Article")

if st.button("Predict"):
    if news.strip() == "":
        st.warning("Please enter a news article to make a prediction.")
    else:
        vec = vectorizer.transform([news])
        pred = model.predict(vec)
        st.success("✅ This news is Real." if pred[0] == "REAL" else "🚨 This news is Fake.")


