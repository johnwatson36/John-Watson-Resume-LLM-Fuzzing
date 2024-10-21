import os
from openai import OpenAI
from typing import Optional, Dict, Any, List
import json  # Import the json module for parsing JSON content
import time

from logging_config import *  # Import the common logging configuration

# Set your API key and base URL for OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",  # Base URL for OpenRouter
    api_key="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"  # Replace with your actual OpenRouter API key
)

def generate_api_calls(
    num_calls: int = 10,  # Default number of API calls to generate
    model: str = "gpt-4",  # Default model to use
    temperature: float = 0.2,  # Temperature of the model - how creative the model is
    max_tokens: Optional[int] = None,  # Maximum number of tokens to generate
    top_p: float = 1.0,  # Top P of the model - controls the randomness of the model
    frequency_penalty: float = 0.0,  # Frequency penalty of the model - controls the frequency of the model
    presence_penalty: float = 0.0,  # Presence penalty of the model - controls the presence of the model
    max_retries: int = 3,  # Maximum number of retries if an error occurs
    retry_delay: float = 2.0,  # Delay in seconds between retries
    **kwargs: Any  # Additional parameters to pass to the model
) -> List[Dict[str, Any]]:
    """
    Generates unique Python API names or calls from the PyTorch library for testing.
    Automatically retries the generation if an error occurs during the process.
    """
    logging.info(f"Starting API generation with {num_calls} calls using model {model} and temperature {temperature}")

    # Prepare the messages for the model
    messages = [
        {"role": "system", "content": "You are an AI coding assistant with a comprehensive understanding of the PyTorch library and its APIs."},
        {"role": "user", "content": f"""You are tasked with generating {num_calls} unique Python API calls from the PyTorch library.

Each API call should:
1. Include a function name from the PyTorch library, especially lesser-known, underused, or complex ones.
2. Include a parameter list with placeholders (e.g., `param1`, `param2`, ...) that can be filled in later for fuzz testing.
3. Represent different categories of operations such as tensor manipulation, mathematical operations, neural network layers, data loading, or GPU-related functionalities.
4. Be relevant for testing the robustness and error handling of PyTorch, especially for edge cases and unusual scenarios.
5. Be distinct, without repeating or overlapping with other generated calls.

Your output should be structured in JSON format inside <apis> tags, with each API call having the following fields:
- `api_name`: The full name of the PyTorch API.
- `parameters`: A list of parameter names (e.g., `param1`, `param2`, ...)

Please generate exactly {num_calls} unique and robust PyTorch API calls. Output your response in JSON format inside `<apis>` tags.
"""}
    ]

    # Prepare the API call parameters
    params: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        **kwargs
    }
    if max_tokens is not None:
        params["max_tokens"] = max_tokens

    # Retry mechanism
    for attempt in range(1, max_retries + 1):
        try:
            logging.info(f"Sending request to OpenRouter API (Attempt {attempt}/{max_retries})...")
            response = client.chat.completions.create(**params)

            # Extract the content from the response
            content = response.choices[0].message.content
            logging.info("Response received successfully.")

            # Check if there is an <apis> tag in the response
            if '<apis>' in content:
                json_content = content.split('<apis>')[1].split('</apis>')[0].strip()
                api_calls = json.loads(json_content)
                logging.info(f"Generated {len(api_calls)} API calls successfully.")
                return api_calls  # Success, return the API calls
            else:
                logging.error("No <apis> tag found in the response.")
                raise ValueError("No <apis> tag found in the response.")

        except json.JSONDecodeError as e:
            logging.error(f"Error during API generation: Invalid JSON structure (Attempt {attempt}/{max_retries}): {str(e)}")
        except Exception as e:
            logging.error(f"Error during API generation (Attempt {attempt}/{max_retries}): {str(e)}")

        # If not successful and retries are left, wait before retrying
        if attempt < max_retries:
            logging.info(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
    
    logging.error("Failed to generate API calls after maximum retries.")
    return [f"An error occurred: Failed after {max_retries} attempts."]  # Return a message if all attempts fail