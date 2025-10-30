import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("📰 Fake News Detector")
st.write("Enter a news headline or article text below and find out if it's real or fake!")

# User input
user_input = st.text_area("🧾 Enter news text here:")

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        data = vectorizer.transform([user_input])
        prediction = model.predict(data)[0]
        if prediction == "FAKE":
            st.error("🚨 This news seems **FAKE**!")
        else:
            st.success("✅ This news seems **REAL**!")
