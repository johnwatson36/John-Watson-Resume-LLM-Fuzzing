import csv
import os  # Import the os module to check for file existence
import json  # Import json module to handle the JSON format output
import temperature_calculator # Import temperature_cal file
from api_gen import generate_api_calls  # Import the function from api_gen.py
from typing import Optional, Dict, Any, List  # Import statement

def save_apis_to_csv(api_calls: List[Dict[str, Any]], filename: str):
    """
    Save the generated API calls to a CSV file.
    """
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['api_name', 'parameters', 'parameter_descriptions', 'example_usage', 'temperature']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for api_call in api_calls:
            writer.writerow(api_call)
    print(f"API calls saved to {filename}")

def get_available_filename(base_filename: str) -> str:
    """
    Generate an available filename by adding a number suffix if needed.
    """
    filename, ext = os.path.splitext(base_filename)
    counter = 1
    new_filename = base_filename
    while os.path.exists(new_filename):
        new_filename = f"{filename}{counter}{ext}"
        counter += 1
    return new_filename

def read_optimal_temperature(filename: str = 'optimal_temperature.txt') -> float:
    """
    Read the optimal temperature from a file.

    Args:
    - filename (str): The file containing the optimal temperature.

    Returns:
    - float: The optimal temperature value.
    """
    try:
        with open(filename, 'r') as temp_file:
            temperature = float(temp_file.read().strip())
        print(f"Read optimal temperature: {temperature}")
        return temperature
    except Exception as e:
        print(f"Error reading optimal temperature: {e}")
        return 0.7  # Default temperature if reading fails

if __name__ == "__main__":
    # Calculate the optimal temperature
    temperature_calculator.calculate_optimal_temperature()

    # Read the optimal temperature from the file
    optimal_temperature = read_optimal_temperature()

    # Generate PyTorch API calls with the optimal temperature
    api_calls = generate_api_calls(num_calls=10, model="gpt-4", temperature=optimal_temperature)
    
    # Check if API calls were generated successfully
    if isinstance(api_calls, list) and api_calls and "An error occurred" not in api_calls[0]:
        # Determine the filename to use
        filename = get_available_filename('apis.csv')
        
        # Save the generated API calls to a CSV file
        save_apis_to_csv(api_calls, filename)
    else:
        print("Failed to generate API calls.")
