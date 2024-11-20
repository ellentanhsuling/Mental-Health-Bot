import streamlit as st
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load the pre-trained model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Set up the chatbot UI
st.title("Mental Health Chatbot")
st.subheader("Talk to us, we're here to help")

input_text_area = st.text_area("Enter your thoughts and feelings...")
button = st.button("Submit")

if button:
    # Process user input
    input_text = input_text_area.strip()
    input_encoding = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model(input_encoding)

    # Get the predicted label (0: normal, 1: urgent help needed)
    predicted_label = torch.argmax(outputs.logits).item()

    if predicted_label == 1:
        st.warning("Urgent attention required! Please seek professional help or talk to someone you trust.")
    else:
        st.info("Thank you for sharing your thoughts. Remember, mental health is just as important as physical health.")