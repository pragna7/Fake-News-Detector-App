import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📰 Fake News Detector")

news = st.text_area("Enter News Article")
if st.button("Predict"):
    vec = vectorizer.transform([news])
    pred = model.predict(vec)
    st.success("This news is Real." if pred[0] == "REAL" else "This news is Fake.")
