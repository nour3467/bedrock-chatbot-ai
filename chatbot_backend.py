"""
Author: Noureddine E.
Date: 09.09.2024
Description: This code is for invoking Bedrock model with LangChain and Streamlit.
"""


# -- Imports: os, Langchain, Memory, ConversationChain
import os
from langchain.llms.bedrock import Bedrock
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain


def demo_chatbot():
    """
    This function initializes the Bedrock client with the specified profile, model ID, and inference parameters.
    Returns a Bedrock client object.

    :return: A Bedrock client object.
    """
    demo_llm = Bedrock(
       credentials_profile_name='default',
       model_id='meta.llama2-70b-chat-v1',
       model_kwargs= {
        "temperature": 0.9,
        "top_p": 0.5,
        "max_gen_len": 512})
    return demo_llm

# -- Test out the LLM with Predict method
#    return demo_llm.predict(input_text)
# response = demo_chatbot('what is the temprature in london like ?')
# print(response)

# -- Create a Function for Conversation Buffer Memory (llm and max token limit)
def demo_memory():
    """
    This function creates a ConversationBufferMemory object with the specified LLM and maximum token limit.

    :return: A ConversationBufferMemory object.
    """
    llm_data=demo_chatbot()
    memory = ConversationBufferMemory(llm=llm_data, max_token_limit= 512)
    return memory

# -- Create a Function for Conversation Chain - Input text + Memory
def demo_conversation(input_text,memory):
    """
    This function creates a ConversationChain object with the specified LLM, memory, and verbose mode.

    :param input_text: The input text for the conversation.
    :param memory: The ConversationBufferMemory object.
    :return: A ConversationChain object.
    """
    llm_chain_data = demo_chatbot()
    llm_conversation= ConversationChain(llm=llm_chain_data,memory= memory,verbose=True)

    # -- Test out the LLM with Predict method
    chat_reply = llm_conversation.predict(input=input_text)
    return chat_reply

