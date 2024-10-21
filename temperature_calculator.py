import csv
import os  # Import the os module to check for file existence
import json  # Import json module to handle the JSON format output
from api_gen import generate_api_calls  # Import the function from api_gen.py
from typing import Optional, Dict, Any, List  # Import statement

from logging_config import *  # Import the common logging configuration

def evaluate_api_quality(api_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Evaluate the quality of generated API calls and assign a score.

    Args:
    - api_calls (List[Dict[str, Any]]): The generated API calls.

    Returns:
    - List[Dict[str, Any]]: The API calls with an added 'score' field for quality.
    """
    for api_call in api_calls:
        score = 0
        num_params = len(api_call.get('parameters', []))
        score += num_params  # Use number of parameters as part of the score

        if len(set(api_call.get('parameters', []))) == len(api_call.get('parameters', [])):  # Unique param names
            score += 1

        if "torch." in api_call.get('api_name', ''):  # Check for valid PyTorch API
            score += 1

        # Add the score to the API call
        api_call['score'] = score
        api_call['num_parameters'] = num_params  # Track number of parameters

    logging.info(f"Evaluated API quality for {len(api_calls)} calls.")
    return api_calls

def calculate_average_score(api_calls: List[Dict[str, Any]]) -> float:
    """
    Calculate the average score of the generated API calls.
    """
    api_calls = evaluate_api_quality(api_calls)
    avg_score = sum(api_call['score'] for api_call in api_calls) / len(api_calls) if api_calls else 0
    logging.info(f"Average API score calculated: {avg_score}")
    return avg_score

def adjust_temperature(current_temperature: float, avg_score: float) -> float:
    """
    Adjusts the temperature based on the average score of generated API calls.
    """
    if avg_score >= 7:
        new_temperature = current_temperature - 0.05
    elif avg_score >= 6.0:
        new_temperature = current_temperature - 0.1
    elif avg_score >= 5.0:
        new_temperature = current_temperature - 0.15
    else:
        new_temperature = current_temperature - 0.2

    logging.info(f"Temperature adjusted from {current_temperature:.2f} to {new_temperature:.2f} based on average score {avg_score:.2f}.")
    return new_temperature

def calculate_optimal_temperature(num_calls):
    """
    Calculate the optimal temperature for generating APIs.
    """
    temperature = 1.0  # Start at 1.0
    results_filename = 'api_results_temp.csv'
    best_temperature = temperature
    highest_avg_score = 0

    # Initialize CSV with headers for logging API call details
    with open(results_filename, 'w', newline='') as csvfile:
        fieldnames = ['api_name', 'parameters', 'temperature', 'score', 'num_parameters']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    logging.info("Starting temperature optimization process...")

    while temperature >= 0.0:  # Loop until temperature reaches 0.0
        logging.info(f"\n--- Round with Temperature {temperature:.2f} ---")
        api_calls = generate_api_calls(num_calls, model="nousresearch/hermes-3-llama-3.1-405b:free", temperature=temperature)
        
        if isinstance(api_calls, list) and api_calls and "An error occurred" not in api_calls[0]:
            avg_score = calculate_average_score(api_calls)

            # Log the generated APIs to CSV
            with open(results_filename, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                for api_call in api_calls:
                    api_call['temperature'] = temperature
                    writer.writerow(api_call)

            # Track the best temperature
            if avg_score > highest_avg_score:
                highest_avg_score = avg_score
                best_temperature = temperature

            # Adjust temperature based on the average score
            new_temperature = adjust_temperature(temperature, avg_score)
            temperature = new_temperature
        else:
            logging.error("Failed to generate API calls.")
            break

    with open('optimal_temperature.txt', 'w') as temp_file:
        temp_file.write(str(best_temperature))
    
    logging.info(f"Optimal temperature calculated: {best_temperature:.2f} with Highest Average Score: {highest_avg_score:.2f}")