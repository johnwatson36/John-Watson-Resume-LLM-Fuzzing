import csv
import os  # Import the os module to check for file existence
import json  # Import json module to handle the JSON format output
from api_gen import generate_api_calls  # Import the function from api_gen.py
from typing import Optional, Dict, Any, List  # Import statement

def evaluate_api_quality(api_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Evaluate the quality of generated API calls and assign a score.

    Args:
    - api_calls (List[Dict[str, Any]]): The generated API calls.

    Returns:
    - List[Dict[str, Any]]: The API calls with an added 'score' field for quality.
    """
    for api_call in api_calls:
        # Initialize the score
        score = 0

        # Scoring criteria based on the number of parameters
        num_params = len(api_call.get('parameters', []))
        score += num_params  # Directly use the number of parameters as the score
        
        if len(set(api_call.get('parameters', []))) == len(api_call.get('parameters', [])):  # Reward unique parameter names
            score += 1
        
        if "torch." in api_call.get('api_name', ''):  # Ensure it's a valid PyTorch API
            score += 1

        # Add the score to the API call
        api_call['score'] = score

    return api_calls

def calculate_average_score(api_calls: List[Dict[str, Any]]) -> float:
    """
    Calculate the average score of the generated API calls.

    Args:
    - api_calls (List[Dict[str, Any]]): The generated API calls.

    Returns:
    - float: The average score.
    """
    api_calls = evaluate_api_quality(api_calls)
    avg_score = sum(api_call['score'] for api_call in api_calls) / len(api_calls) if len(api_calls) > 0 else 0
    return avg_score

def adjust_temperature(current_temperature: float, avg_score: float) -> float:
    """
    Adjusts the temperature based on the average score of generated API calls.

    Args:
    - current_temperature (float): The current temperature value.
    - avg_score (float): The average score of the generated API calls.

    Returns:
    - float: The adjusted temperature value.
    """
    # Adjust temperature based on the average score
    if avg_score >= 7:  # High quality, reduce temperature minimally
        new_temperature = current_temperature - 0.05  # Allow going below 0.0
    elif avg_score >= 6.0:  # Moderate quality, reduce temperature slightly
        new_temperature = current_temperature - 0.1  # Allow going below 0.0
    elif avg_score >= 5.0:  # Low to moderate quality, reduce temperature more
        new_temperature = current_temperature - 0.15  # Allow going below 0.0
    else:  # Poor quality, reduce temperature significantly
        new_temperature = current_temperature - 0.2  # Allow going below 0.0

    return new_temperature

def calculate_optimal_temperature():
    """
    Calculate the optimal temperature for generating APIs.
    """
    # Initialize variables
    temperature = 1.0  # Start at 1.0
    results_filename = 'api_results_temp.csv'  # Intermediate results for analysis
    best_temperature = temperature
    highest_avg_score = 0

    # Initialize CSV with headers for logging
    with open(results_filename, 'w', newline='') as csvfile:
        fieldnames = ['api_name', 'parameters', 'parameter_descriptions', 'example_usage', 'temperature', 'score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    # Loop through a few rounds of API generation and temperature adjustment
    while temperature >= 0.0:  # Continue until the temperature is reduced to 0.0
        print(f"\n--- Round with Temperature {temperature:.2f} ---")

        # Generate PyTorch API calls with the current temperature
        api_calls = generate_api_calls(num_calls=10, model="gpt-4", temperature=temperature)
        
        # Check if API calls were generated successfully
        if isinstance(api_calls, list) and api_calls and "An error occurred" not in api_calls[0]:
            # Evaluate the API quality and calculate the average score
            avg_score = calculate_average_score(api_calls)

            # Log the generated APIs to CSV
            with open(results_filename, 'a', newline='') as csvfile:
                fieldnames = ['api_name', 'parameters', 'parameter_descriptions', 'example_usage', 'temperature', 'score']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                for api_call in api_calls:
                    api_call['temperature'] = temperature  # Add temperature info to each API result
                    writer.writerow(api_call)
            
            # Keep track of the best temperature with the highest average score
            if avg_score > highest_avg_score:
                highest_avg_score = avg_score
                best_temperature = temperature  # Update best_temperature before changing temperature

            # Adjust temperature based on the generated APIs
            new_temperature = adjust_temperature(temperature, avg_score)
            print(f"Adjusted temperature from {temperature:.2f} to {new_temperature:.2f} based on average score {avg_score:.2f}")
            temperature = new_temperature

        else:
            print("Failed to generate API calls.")
            break

    # Save the final optimal temperature to a file
    with open('optimal_temperature.txt', 'w') as temp_file:
        temp_file.write(str(best_temperature))
    print(f"Optimal temperature calculated and saved: {best_temperature:.2f} with Highest Average Score: {highest_avg_score:.2f}")

# if __name__ == "__main__":
#     calculate_optimal_temperature()
