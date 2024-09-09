"""
🤖 NourBot: Your AI Companion 🚀
Author: Noureddine E.
Date: 09.09.2024
Source: Based on code provided by Streamlit and AWS, customized for awesomeness!
"""

import streamlit as st
import chatbot_backend as demo

# Set a cool title for the chatbot
st.title("Meet NourBot: Your AI Sidekick 😎🚀")

# Initialize LangChain memory in the session state
if "memory" not in st.session_state:
    st.session_state.memory = demo.demo_memory()

# Initialize chat history in the session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Re-render the chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["text"])

# Create an awesome chat input box
input_text = st.chat_input("💬 Chat with NourBot (Powered by Bedrock and LLama 2)")

if input_text:
    # Display user message
    with st.chat_message("user"):
        st.markdown(input_text)
    st.session_state.chat_history.append({"role": "user", "text": input_text})

    # Generate and display AI response
    chat_response = demo.demo_conversation(
        input_text=input_text, memory=st.session_state.memory
    )
    with st.chat_message("assistant"):
        st.markdown(chat_response)
    st.session_state.chat_history.append({"role": "assistant", "text": chat_response})

# Add a cool footer
st.markdown("---")
st.markdown(
    "👨‍💻 Crafted with ❤️ by Noureddine E. | 🔗 [Connect on LinkedIn](www.linkedin.com/in/nour3467)"
)
