import streamlit as st
import pickle
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import time

nltk.download("punkt_tab")
nltk.download("stopwords")

ps = PorterStemmer()


def transform_text(text):

    tokens = nltk.word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [
        ps.stem(token)
        for token in tokens
        if token.isalnum() and token not in stop_words
    ]

    return " ".join(filtered_tokens)


# Load vectorizer and model once
@st.cache_resource
def load_model_and_vectorizer():
    vectorizer = pickle.load(open("vector.pkl", "rb"))
    model = pickle.load(open("model.pkl", "rb"))
    return vectorizer, model


vectorizer, model = load_model_and_vectorizer()
st.image("./png.png", width=200)
st.markdown(
    "<h1 style='text-align: center;'>SMS SPAM 💌 DETECTOR</h1>",
    unsafe_allow_html=True,
)


def reset_input():
    st.session_state.input_sms = ""


# Input from user
input_sms = st.text_input(
    "", placeholder="Enter your SMS to check if it's spam:", key="input_sms"
)

if st.button("Analyze SMS"):
    if not input_sms.strip():
        st.warning("Please enter an SMS before analyzing!")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = vectorizer.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        # Display the result
        if result == 0:
            st.header("✔️ This SMS is NOT Spam!")
        else:
            st.header("⚠️ Warning: This SMS is Spam!")

    reset_input()
