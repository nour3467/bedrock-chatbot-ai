"""
🤖 NourBot: Your AI Companion 🚀
Author: Noureddine E.
Date: 09.09.2024
Description: This code is for testing Bedrock model with LangChain and Streamlit.
"""
import os
import boto3
import json
import base64
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# -- Fetch credentials and region from the environment
ACCESS_KEY = os.getenv("ACCESS_KEY")
SECRET_ACCESS_KEY = os.getenv("SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")


def create_bedrock_runtime_client():
    """
    Create and return the Bedrock runtime client.
    """
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


def invoke_bedrock_model_via_messages_api(
    client, model_id, messages, model_kwargs, anthropic_version=None, max_tokens=1000
):
    """
    Invoke the Bedrock model using the Messages API, handling both text and images.
    """
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
        body.update(model_kwargs)

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


def setup_llm(
    model_id="anthropic.claude-instant-v1",
    anthropic_version=None,
    max_tokens=1000,
    temperature=0.0,
    top_k=250,
    top_p=1,
):
    """
    Setup the LLM configuration and return a dictionary with the model ID, parameters, and runtime client.
    This function supports both text and image inputs and adjusts parameters based on the model.
    """
    client = create_bedrock_runtime_client()
    if client is None:
        raise ValueError("Could not create Bedrock runtime client")

    model_params = {"temperature": temperature, "top_k": top_k, "top_p": top_p}

    return {
        "model_id": model_id,
        "model_kwargs": model_params,
        "client": client,
        "max_tokens": max_tokens,
        "anthropic_version": anthropic_version,  # Set anthropic_version if needed
    }


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

    # Add image input if provided
    if base64_image:
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


if __name__ == "__main__":
    # Example 1: Run with a text model (Messages API)
    llm_config = setup_llm(
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        anthropic_version="bedrock-2023-05-31",
    )
    result = run_llm_inference(
        input_text="What is the capital of France?", model_config=llm_config
    )
    print(result["response_text"])  # Print the assistant's response

    # Example 2: Run with both image and text input (Messages API)
    # get image from images/image_1.png
    with open("./images/image_1.png", "rb") as image_file:
        base64_image_data = base64.b64encode(image_file.read()).decode("utf-8")

    llm_config_image = setup_llm(
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        anthropic_version="bedrock-2023-05-31",
    )
    result_image = run_llm_inference(
        input_text="What's in this image?",
        base64_image=base64_image_data,
        model_config=llm_config_image,
    )
    print(result_image["response_text"])  # Print the assistant's response
