# app/services/llama_service.py
import os
import replicate
from flask import current_app
from dotenv import load_dotenv

load_dotenv()

def get_llama_model(model_name):
    """
    Get the appropriate model ID based on the model name.
    
    Args:
        model_name (str): Name of the model (e.g., 'Llama2-7B', 'Llama2-13B')
    
    Returns:
        str: The model ID for Replicate API
    """
    models = {
        'Llama2-7B': 'a16z-infra/llama7b-v2-chat:4f0a4744c7295c024a1de15e1a63c880d3da035fa1f49bfd344fe076074c8eea',
        'Llama2-13B': 'a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5'
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available models: {', '.join(models.keys())}")
    
    return models[model_name]

def generate_llama_response(prompt, model_name, temperature=0.2, top_p=0.9, max_length=100):
    """
    Generate a response using the LLaMA model via Replicate API.
    
    Args:
        prompt (str): The input text prompt
        model_name (str): The name of the model (e.g., "Llama2-7B")
        temperature (float): Controls randomness in generation (0-1)
        top_p (float): Controls diversity of generation (0-1)
        max_length (int): Maximum length of generated response
        
    Returns:
        str: Generated response text
    """
    # First try to get API key from Flask config
    api_token = current_app.config.get('REPLICATE_API_KEY')
     
    # If not in Flask config, try environment variable
    if not api_token:
        api_token = os.getenv('REPLICATE_API_TOKEN')
    
    # If still no API token, raise error
    if not api_token:
        raise ValueError("No Replicate API token found in either Flask config or environment variables")
    
    # Initialize client with API token
    client = replicate.Client(api_token=api_token)
    
    # Set up the conversation context
    string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
    formatted_prompt = f"{string_dialogue} User: {prompt} Assistant: "
    
    try:
        # Get the correct model ID
        model_id = get_llama_model(model_name)
        
        # Run the model
        output = client.run(
            model_id,
            input={
                "prompt": formatted_prompt,
                "temperature": temperature,
                "top_p": top_p,
                "max_length": max_length,
                "repetition_penalty": 1
            }
        )
        
        # Join the output if it's a sequence
        return ''.join(output) if isinstance(output, (list, tuple)) else output
        
    except Exception as e:
        # Add more context to the error
        error_msg = str(e)
        if "401" in error_msg:
            raise Exception(
                "Authentication failed with Replicate API. Please check your API token. "
                "Make sure REPLICATE_API_KEY is set in Flask config or "
                "REPLICATE_API_TOKEN is set in environment variables."
            )
        raise Exception(f"Failed to generate response: {error_msg}")

def init_llama(app):
    """
    Initialize Llama service with Flask app context.
    Can be used to set up any necessary configuration.
    """
    # You can add any initialization code here if needed in the future
    pass