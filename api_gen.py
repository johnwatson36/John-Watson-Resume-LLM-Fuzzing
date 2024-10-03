import os
from openai import OpenAI
from typing import Optional, Dict, Any, List
import json  # Import the json module for parsing JSON content

client = OpenAI(api_key="my api code")

def generate_api_calls(
    num_calls: int = 10,  # Number of API calls to generate
    model: str = "gpt-4",  # Model to use
    temperature: float = 0.7,  # Temperature of the model - how creative the model is
    max_tokens: Optional[int] = None,  # Maximum number of tokens to generate
    top_p: float = 1.0,  # Top P of the model - controls the randomness of the model
    frequency_penalty: float = 0.0,  # Frequency penalty of the model - controls the frequency of the model
    presence_penalty: float = 0.0,  # Presence penalty of the model - controls the presence of the model
    **kwargs: Any  # Additional parameters to pass to the model
) -> List[Dict[str, Any]]:
    """
    Generates unique Python API names or calls from the PyTorch library for testing.
    """
    # Prepare the messages for the model
    messages = [
        {"role": "system", "content": "You are an AI coding assistant with a comprehensive understanding of the PyTorch library and its APIs."},
        {"role": "user", "content": f"""You are tasked with generating {num_calls} unique Python API calls from the PyTorch library.

Each API call should:
1. Include a function name from the PyTorch library, especially lesser-known, underused, or complex ones.
2. Include a parameter list with placeholders (e.g., `param1`, `param2`, ...) that can be filled in later for fuzz testing.
3. Provide a brief description of what each parameter represents (e.g., "Number of elements", "Tensor size", "Data type").
4. Represent different categories of operations such as tensor manipulation, mathematical operations, neural network layers, data loading, or GPU-related functionalities.
5. Be relevant for testing the robustness and error handling of PyTorch, especially for edge cases and unusual scenarios.
6. Be distinct, without repeating or overlapping with other generated calls.

Your output should be structured in JSON format inside <apis> tags, with each API call having the following fields:
- `api_name`: The full name of the PyTorch API.
- `parameters`: A list of parameter names (e.g., `param1`, `param2`, ...)
- `parameter_descriptions`: A list of descriptions for each parameter.
- `example_usage`: A short example of the API call using the placeholders.

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
    # Add max_tokens if provided
    if max_tokens is not None:
        params["max_tokens"] = max_tokens

    try:
        # Make the API call using OpenAI library
        response = client.chat.completions.create(**params)
        print("Response: ", response)

        # Extract the content from the response
        content = response.choices[0].message.content
        print("Generated API Calls:\n", content)

        # Check if there is an <apis> tag in the response
        if '<apis>' in content:
            # Extract the JSON content from the response
            json_content = content.split('<apis>')[1].split('</apis>')[0].strip()
            # Parse the JSON content
            api_calls = json.loads(json_content)
        else:
            return [f"An error occurred: No <apis> tag found in the response."]

        return api_calls

    except Exception as e:
        print("Error: ", e)
        return [f"An error occurred: {str(e)}"]
