import streamlit as st
import google.generativeai as genai

# Configure Gemini API
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel('gemini-pro')

def get_bot_response(prompt):
    response = model.generate_content(
        f"""As a mental health support assistant, respond to: {prompt}
        Remember to be empathetic and supportive while maintaining appropriate boundaries."""
    )
    return response.text

# Set up the chatbot UI
st.title("Mental Health Support Bot")
st.subheader("Let's have a supportive conversation")

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for message in st.session_state.chat_history:
    st.write(message)

# Get user input
user_input = st.text_input("Share what's on your mind...")

if user_input:
    # Add user message to chat history
    st.session_state.chat_history.append(f"You: {user_input}")
    
    # Get bot response using Gemini API
    bot_response = get_bot_response(user_input)
    st.session_state.chat_history.append(f"Bot: {bot_response}")
    
    # Clear the input field
    st.empty()

# Add helpful resources
st.sidebar.title("Support Resources")
st.sidebar.write("24/7 Crisis Support:")
st.sidebar.write("- National Crisis Hotline: 988")
st.sidebar.write("- Crisis Text Line: Text HOME to 741741")
st.sidebar.write("- Emergency: 911")
