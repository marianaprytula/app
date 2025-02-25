import streamlit as st
from transformers import pipeline
import os

# Load the Hugging Face token from Streamlit secrets
HUGGINGFACE_TOKEN = st.secrets["HUGGINGFACE_TOKEN"]  # Store this in `.streamlit/secrets.toml`

# Load the model pipeline with authentication
@st.cache_resource
def load_pipeline():
    return pipeline(
        ""text-generation"",
        model="Marivanna27/fine-tuned-model_llama3_1_binary",
        token=HUGGINGFACE_TOKEN  # Use the token for authentication
    )

classifier = load_pipeline()

# Streamlit UI
st.title("LLM Text Classification App")
st.write("Enter text below, and the model will classify it.")

user_input = st.text_area("Enter your text:", "")

if st.button("Classify"):
    if user_input:
        prediction = classifier(user_input)
        st.write(f"**Prediction:** {prediction[0]['label']}")
        st.write(f"**Confidence:** {prediction[0]['score']:.4f}")
    else:
        st.warning("Please enter some text for classification.")