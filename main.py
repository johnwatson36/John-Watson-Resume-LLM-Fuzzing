import csv
import os
import json
import temperature_calculator
from api_gen import generate_api_calls
from typing import Optional, Dict, Any, List
import pytorch_bug_analyzer
import graphs
from logging_config import *
from datetime import datetime  # For calculating the runtime

def save_apis_to_csv(api_calls: List[Dict[str, Any]], filename: str):
    if os.path.exists(filename):
        os.remove(filename)
        logging.info(f"Existing file {filename} deleted.")

    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['api_name', 'parameters']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for api_call in api_calls:
            writer.writerow({
                'api_name': api_call.get('api_name', ''),
                'parameters': ', '.join(api_call.get('parameters', []))
            })
    
    logging.info(f"API calls saved to {filename}")

def read_optimal_temperature(filename: str = 'optimal_temperature.txt') -> float:
    try:
        with open(filename, 'r') as temp_file:
            temperature = float(temp_file.read().strip())
        logging.info(f"Read optimal temperature: {temperature}")
        return temperature
    except Exception as e:
        logging.error(f"Error reading optimal temperature: {e}")
        return 0.7  # Default temperature if reading fails

if __name__ == "__main__":
    try:
        # Start measuring the total runtime
        total_start_time = datetime.now()
        logging.info("Starting the entire program...")

        # Calculate the optimal temperature
        temperature_calculator.calculate_optimal_temperature(num_calls=10)

        # Read the optimal temperature from the file
        optimal_temperature = read_optimal_temperature()
        logging.info(f"Optimal temperature for API generation: {optimal_temperature}")

        # Generate PyTorch API calls with the optimal temperature
        api_calls = generate_api_calls(num_calls=10, model="nousresearch/hermes-3-llama-3.1-405b:free", temperature=optimal_temperature)

        # Check if API calls were generated successfully
        if isinstance(api_calls, list) and api_calls and "An error occurred" not in api_calls[0]:
            save_apis_to_csv(api_calls, 'apis.csv')
        else:
            logging.error("Failed to generate API calls.")

        # Call graphs and assess the APIs
        graphs.create_graphs()

        # Call pytorch_bug_analyzer's main function
        pytorch_bug_analyzer.main()

        # End measuring the total runtime
        total_end_time = datetime.now()
        total_runtime = total_end_time - total_start_time
        logging.info(f"Total runtime of the entire program: {total_runtime}")

    except Exception as e:
        logging.critical(f"Error occurred in the main script: {e}")
