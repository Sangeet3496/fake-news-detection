import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("📰 Fake News Detection App")

news = st.text_area("Enter News Text")

if st.button("Predict"):
    if news.strip() == "":
        st.warning("Please enter some text")
    else:
        news_vec = vectorizer.transform([news])
        prediction = model.predict(news_vec)

        if prediction[0] == 1:
            st.success("REAL NEWS 🟢")
        else:
            st.error("FAKE NEWS 🔴")