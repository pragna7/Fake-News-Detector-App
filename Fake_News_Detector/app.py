import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📰 Fake News Detector")

# Input area for news
news = st.text_area("Enter News Article")

# Predict button
if st.button("Predict"):
    if news.strip() == "":
        st.warning("Please enter a news article to make a prediction.")
    else:
        vec = vectorizer.transform([news])
        pred = model.predict(vec)
        st.success("✅ This news is Real." if pred[0] == "REAL" else "🚨 This news is Fake.")

# Show model accuracy button
if st.button("Show Model Accuracy"):
    try:
        with open("accuracy.txt", "r") as f:
            accuracy = f.read()
            st.info(f"📊 Model Accuracy: {float(accuracy) * 100:.2f}%")
    except FileNotFoundError:
        st.error("⚠️ Accuracy file not found. Please make sure 'accuracy.txt' is in the same folder.")

