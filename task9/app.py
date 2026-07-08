import streamlit as st
import pickle
from textblob import TextBlob

# Load model
model = pickle.load(
    open(
        "sentiment_model.pkl",
        "rb"
    )
)

vectorizer = pickle.load(
    open(
        "vectorizer.pkl",
        "rb"
    )
)

# Page settings
st.set_page_config(
    page_title=
    "AI-Based Sentiment Analysis",
    layout="centered"
)

# Title
st.title(
    "AI-Based Sentiment Analysis for Social Media"
)

st.write(
    "Analyze and classify social media sentiments"
)

# Text input
user_text = st.text_area(
    "Enter Social Media Text"
)

# Button
if st.button(
    "Analyze Sentiment"
):

    if user_text.strip() == "":
        st.warning(
            "Please enter text"
        )

    else:

        # ML Prediction
        transformed_text = (
            vectorizer.transform(
                [user_text]
            )
        )

        prediction = (
            model.predict(
                transformed_text
            )[0]
        )

        # NLP Polarity
        polarity = (
            TextBlob(
                user_text
            ).sentiment.polarity
        )

        st.subheader(
            "Prediction"
        )

        # Final output
        if prediction == "positive":
            st.success(
                "😊 Positive Sentiment"
            )

        elif prediction == "negative":
            st.error(
                "😞 Negative Sentiment"
            )

        else:
            st.warning(
                "😐 Neutral Sentiment"
            )

        st.write(
            f"Polarity Score: "
            f"{polarity:.2f}"
        )