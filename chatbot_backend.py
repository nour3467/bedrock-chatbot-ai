"""
Author: Noureddine E.
Date: 09.09.2024
Description: This code is for invoking Bedrock model with LangChain and Streamlit.
"""


import os
import json
import base64
import shutup
from dotenv import load_dotenv
from langchain.llms.bedrock import Bedrock
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# ignore DeprecationWarning
shutup.please()

# Load environment variables from .env
load_dotenv()

# Fetch credentials and region from environment
ACCESS_KEY = os.getenv("ACCESS_KEY")
SECRET_ACCESS_KEY = os.getenv("SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")


# -- Utility function for Bedrock client creation
def create_bedrock_runtime_client():
    """
    Create and return the Bedrock runtime client.
    """
    import boto3

    try:
        session = boto3.Session(
            region_name=AWS_REGION,
            aws_access_key_id=ACCESS_KEY,
            aws_secret_access_key=SECRET_ACCESS_KEY,
        )
        return session.client("bedrock-runtime")
    except Exception as e:
        print(f"Error creating Bedrock runtime client: {e}")
        return None


# Function to handle the interaction with the Bedrock model (handling both text and images)
def invoke_bedrock_model_via_messages_api(
    client, model_id, messages, model_kwargs, anthropic_version=None, max_tokens=1000
):
    try:
        # Prepare the body of the request with the messages and additional model parameters
        body = {
            "messages": messages,  # List of messages with "role" and "content"
            "max_tokens": max_tokens,
        }

        # Add model-specific parameters
        if anthropic_version:
            body["anthropic_version"] = anthropic_version

        # Merge with other model-specific parameters
        body |= model_kwargs

        # Call the Bedrock runtime client to invoke the model via the Messages API
        response = client.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body),
        )

        # Decode the response and parse the JSON
        response_body = response["body"].read().decode("utf-8")
        response_json = json.loads(response_body)

        # Extract the 'content' from the response
        assistant_content = response_json.get("content", [])
        output_text = "".join(
            content_item["text"] + "\n"
            for content_item in assistant_content
            if content_item["type"] == "text"
        )
        return {
            "response_text": output_text.strip(),  # Clean up any extra newlines
            "full_response": response_json,  # Return the entire response for additional context
        }

    except Exception as e:
        print(f"Error invoking the model via Messages API: {e}")
        return None


# Function to initialize LLM configuration
def setup_llm(
    model_id="anthropic.claude-3-haiku-20240307-v1:0",
    anthropic_version="bedrock-2023-05-31",
    max_tokens=1000,
):
    client = create_bedrock_runtime_client()
    if client is None:
        raise ValueError("Could not create Bedrock runtime client")

    model_params = {
        "temperature": 0.7,
        "top_k": 250,
        "top_p": 1,
    }

    return {
        "model_id": model_id,
        "model_kwargs": model_params,
        "client": client,
        "max_tokens": max_tokens,
        "anthropic_version": anthropic_version,  # Set anthropic_version if needed
    }


# Chatbot function to handle text and images
def run_llm_inference(input_text=None, base64_image=None, model_config=None):
    """
    Run the model inference by invoking the LLM with the given input and configuration.
    Supports both text and image inputs (as base64).
    """
    client = model_config["client"]
    model_id = model_config["model_id"]
    model_kwargs = model_config["model_kwargs"]
    max_tokens = model_config["max_tokens"]
    anthropic_version = model_config.get("anthropic_version", None)

    # Prepare the messages for the Messages API
    messages = []

    # Add text input if provided
    if input_text:
        messages.append(
            {"role": "user", "content": [{"type": "text", "text": input_text}]}
        )

    # Add image input if provided, create a message entry if no text was provided
    if base64_image:
        if not messages:
            # If no text was provided, create a message with only the image
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64_image,
                            },
                        }
                    ],
                }
            )
        else:
            # Append image to the existing message if there was text
            messages[-1]["content"].append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64_image,
                    },
                }
            )

    return invoke_bedrock_model_via_messages_api(
        client, model_id, messages, model_kwargs, anthropic_version, max_tokens
    )


# Function to run the chatbot and integrate it with LangChain's memory
def chatbot_with_memory(input_text=None, base64_image=None, memory=None):
    """
    This function integrates Bedrock API with Langchain's memory for context-based conversations.
    :param input_text: The input text or question from the user.
    :param base64_image: Base64-encoded image if any image is part of the conversation.
    :param memory: ConversationBufferMemory to store the context.
    :return: The response from the chatbot.
    """
    # Ensure that input_text is not None when saving to memory
    if input_text is None and base64_image is None:
        raise ValueError("Either input_text or base64_image must be provided.")

    # Setup the model configuration
    llm_config = setup_llm()

    # Get response from the model
    response = run_llm_inference(
        input_text=input_text, base64_image=base64_image, model_config=llm_config
    )

    # Ensure the chatbot response is not None
    if response is None or response["response_text"] == "":
        raise ValueError("The chatbot did not return a valid response.")

    # Save context to memory only if valid input_text and response are present
    if memory is not None and input_text:
        memory.save_context(
            {"user_input": input_text}, {"chatbot_response": response["response_text"]}
        )

    return response["response_text"]


# Initialize memory for conversation history
def initialize_memory():
    """
    This function initializes ConversationBufferMemory for storing conversation history.
    :return: ConversationBufferMemory object
    """
    return ConversationBufferMemory()


# Example usage
if __name__ == "__main__":
    # Initialize conversation memory
    memory = initialize_memory()

    # Test chatbot with text input
    text_response = chatbot_with_memory(
        input_text="What is the capital of France?", memory=memory
    )
    print(text_response)

    # Test chatbot with image and text input
    with open("./images/image_1.png", "rb") as image_file:
        base64_image_data = base64.b64encode(image_file.read()).decode("utf-8")

    image_response = chatbot_with_memory(
        input_text="What's in this image?",
        base64_image=base64_image_data,
        memory=memory,
    )
    print(image_response)
