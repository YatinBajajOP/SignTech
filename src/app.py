import streamlit as st
from functions import *

st.header("Welcome to Signtech")

choices = ["Text to Sign Language", "Sign Language to Text"]

choice = st.sidebar.selectbox("Functionalities", choices)

if choice == "Text to Sign Language":
    st.write("Here the text is converted into sign language.")
    text = st.text_area("Enter the text you want to Convert : ")
    if st.button("Submit"):
        textToSign(text)
else:
    st.write("Here the sign language is converted into text.")
    if st.button("Start Camera"):
        signToText()

