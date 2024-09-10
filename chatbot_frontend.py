"""
🤖 NourBot: Your AI Companion 🚀
Author: Noureddine E.
Date: 09.09.2024
Source: Based on code provided by Streamlit and AWS, customized for awesomeness!
"""


import streamlit as st
import chatbot_backend as demo
import base64

# Set a cool title for the chatbot
st.title("Meet NourBot: Your AI Sidekick 😎🚀")

# Initialize LangChain memory in the session state
if "memory" not in st.session_state:
    st.session_state.memory = demo.initialize_memory()

# Initialize chat history in the session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display the chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["text"])

# Add a text input field for user input
input_text = st.chat_input("💬 Chat with NourBot (Powered by Bedrock and LLama 2)")

# Add an image uploader
uploaded_image = st.file_uploader(
    "Upload an image to analyze (optional)", type=["png", "jpg", "jpeg"]
)

# Process the input when the user submits text or uploads an image
if input_text or uploaded_image:

    # Display user message
    with st.chat_message("user"):
        st.markdown(input_text if input_text else "Uploaded an image for analysis")

    # Add user input to chat history
    st.session_state.chat_history.append({"role": "user", "text": input_text})

    # Convert uploaded image to base64 if provided
    base64_image_data = None
    if uploaded_image:
        base64_image_data = base64.b64encode(uploaded_image.read()).decode("utf-8")

    # Show a spinner animation while waiting for the response
    with st.spinner("NourBot is thinking... 🤔"):
        # Call the backend to process the input (text + optional image)
        chat_response = demo.chatbot_with_memory(
            input_text=input_text,
            base64_image=base64_image_data,
            memory=st.session_state.memory,
        )

    # Display the AI response
    with st.chat_message("assistant"):
        st.markdown(chat_response)

    # Add the AI response to chat history
    st.session_state.chat_history.append({"role": "assistant", "text": chat_response})

# Add a cool footer
st.markdown("---")
st.markdown(
    "👨‍💻 Crafted with ❤️ by Noureddine E. | 🔗 [Connect on LinkedIn](https://www.linkedin.com/in/nour3467)"
)
