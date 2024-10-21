# import os
# import requests
# import re
# import random
# import time
# import json
# import subprocess
# import tempfile
# import sys
# import csv
# from datetime import datetime
# from openai import OpenAI
# import torch

# from logging_config import *  # Import the common logging configuration

# # Set your API key and base URL for OpenRouter
# client = OpenAI(
#     base_url="https://openrouter.ai/api/v1",  # Base URL for OpenRouter
#     api_key="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"  # Your OpenRouter API key
# )

# # Global variables for statistics
# total_programs = 0
# valid_programs = 0
# buggy_programs = 0
# api_coverage = set()
# program_recalls = 0
# start_time = None

# def update_statistics(runtime, log_file="results.txt"):
#     file_exists = os.path.isfile("statistics.csv")
    
#     with open("statistics.csv", "a", newline='') as f:
#         writer = csv.writer(f)
#         if not file_exists:
#             writer.writerow(["Timestamp", "Total programs", "Valid programs", "Buggy programs", "API coverage", "Program recalls", "Runtime (seconds)"])
        
#         row = [
#             datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#             total_programs,
#             valid_programs,
#             buggy_programs,
#             f"{len(api_coverage)}/{len(dir(torch))}",
#             program_recalls,
#             round(runtime.total_seconds(), 2)
#         ]
        
#         writer.writerow(row)
#         logging.info(f"Statistics updated: {row}")  # Log statistics to file

# def save_buggy_program(code, index):
#     if not os.path.exists("buggy-programs"):
#         os.makedirs("buggy-programs")
#     filename = f"buggy_program_{index}.py"
#     clean_code = clean_code_snippet(code)
#     with open(os.path.join("buggy-programs", filename), "w") as f:
#         f.write(clean_code)
#     logging.info(f"Buggy program saved as: {filename}")

# def clean_code_snippet(code):
#     return code.strip()

# # Function to read APIs from a .csv file
# def get_generated_apis(filename="apis.csv"):
#     """
#     Load API calls from a CSV file, ignoring the header.

#     Args:
#     - filename: Name of the CSV file (default: apis.csv)

#     Returns:
#     - List of tuples where each tuple is (api_name, parameters)
#     """
#     api_list = []
#     try:
#         with open(filename, mode='r') as file:
#             csv_reader = csv.reader(file)
#             next(csv_reader)  # Skip the header row
#             for row in csv_reader:
#                 api_name = row[0].strip()  # First column is the API name
#                 parameters = [param.strip() for param in row[1].split(',')]  # Split parameters by commas and trim whitespaces
#                 api_list.append((api_name, parameters))
#         logging.info(f"APIs loaded successfully from {filename}")
#     except FileNotFoundError:
#         logging.error(f"API file {filename} not found.")
#     return api_list

# def analyze_pytorch_bug(api_name, parameters):
#     """
#     Generate a new test program based on the API and parameters.
#     """
#     formatted_parameters = ', '.join(parameters)
#     prompt = f"""
# Generate a runnable Python PyTorch program to test the API {api_name} with the following parameters:
# {formatted_parameters}. 

# The program should:
# - Run without errors.
# - Use the parameters and produce realistic test values (e.g., tensor data).
# - Show potential issues or differences between CPU and GPU.
# - Ensure reproducibility by setting random seeds.

# Provide only the complete code, without any additional explanations.
# """
#     try:
#         response = client.chat.completions.create(
#             model="nousresearch/hermes-3-llama-3.1-405b:free",
#             messages=[
#                 {"role": "system", "content": "You are a PyTorch developer tasked with creating example programs for API testing."},
#                 {"role": "user", "content": prompt}
#             ]
#         )
        
#         if response and response.choices and len(response.choices) > 0 and response.choices[0].message:
#             generated_code = response.choices[0].message.content
#             generated_code = generated_code.replace("```python", "").replace("```", "")
#             generated_code = generated_code.replace('api_name', api_name)
#             for i, param in enumerate(parameters):
#                 generated_code = generated_code.replace(f'param{i+1}', param)
#             logging.info(f"Generated program for API {api_name}")
#             return clean_code_snippet(generated_code)
#         else:
#             logging.error("Error: Unexpected response structure from the API.")
#             return None
#     except Exception as e:
#         logging.error(f"Error occurred while generating new program: {str(e)}")
#         return None

# def execute_python_code(code):
#     global total_programs, valid_programs, buggy_programs, api_coverage

#     clean_code = clean_code_snippet(code)
#     seed = random.randint(0, 2**32 - 1)
#     seeded_code = f"""
# import random
# import numpy as np
# import torch

# random.seed({seed})
# np.random.seed({seed})
# torch.manual_seed({seed})
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all({seed})

# {clean_code}
# """

#     with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
#         temp_file.write(seeded_code)
#         temp_file_path = temp_file.name

#     try:
#         python_executable = sys.executable
#         result_cpu = subprocess.run([python_executable, temp_file_path], capture_output=True, text=True, timeout=10, env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1"))
#         output_cpu = result_cpu.stdout
#         error_cpu = result_cpu.stderr

#         if torch.cuda.is_available():
#             result_gpu = subprocess.run([python_executable, temp_file_path], capture_output=True, text=True, timeout=10, env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"))
#             output_gpu = result_gpu.stdout
#             error_gpu = result_gpu.stderr
#         else:
#             output_gpu = output_cpu
#             error_gpu = error_cpu

#         if "SyntaxError" in error_cpu or "SyntaxError" in error_gpu:
#             logging.error(f"SyntaxError Detected in Program Execution\nCPU Error: {error_cpu}\nGPU Error: {error_gpu}")
#             return None, "Syntax error detected", None, None, "Program has syntax errors."

#         if "TypeError" in error_cpu or "TypeError" in error_gpu:
#             logging.error(f"TypeError Detected in Program Execution\nCPU Error: {error_cpu}\nGPU Error: {error_gpu}")
#             return None, "TypeError detected", None, None, "Program has a TypeError and needs to be regenerated."

#         if output_cpu != output_gpu or error_cpu != error_gpu:
#             buggy_programs += 1
#             save_buggy_program(seeded_code, buggy_programs)
#             explanation = f"Program produced different results on CPU vs GPU."
#             logging.warning(f"CPU Output: {output_cpu}\nCPU Error: {error_cpu}\nGPU Output: {output_gpu}\nGPU Error: {error_gpu}")
#             return output_cpu, error_cpu, output_gpu, error_gpu, explanation

#         valid_programs += 1
#         for line in code.split('\n'):
#             if 'torch.' in line:
#                 api_call = line.split('torch.')[1].split('(')[0]
#                 api_coverage.add(api_call)

#         logging.info("Program executed successfully without bugs.")
#         return output_cpu, error_cpu, output_gpu, error_gpu, "Program executed successfully without bugs."

#     except subprocess.TimeoutExpired:
#         explanation = "Execution timed out after 10 seconds"
#         logging.error(explanation)
#         return "Execution timed out after 10 seconds", "", "", "", explanation
#     except Exception as e:
#         logging.error(f"Error during program execution: {e}")
#         return None, f"Execution failed: {e}", None, None, "Execution failed."
#     finally:
#         os.unlink(temp_file_path)

# def main():
#     global program_recalls, total_programs, valid_programs, buggy_programs, start_time

#     logging.info("Starting Automated PyTorch API Tester...")
#     start_time = datetime.now()

#     apis = get_generated_apis("apis.csv")
    
#     logging.info("Analyzing and generating programs for the APIs listed in apis.csv:")
#     for api_index, (api_name, parameters) in enumerate(apis):
#         total_programs += 1
#         logging.info(f"Program {total_programs} - {api_name} with parameters {parameters}")
        
#         attempts = 0
#         max_attempts = 3
#         while attempts < max_attempts:
#             generated_program = analyze_pytorch_bug(api_name, parameters)
#             if generated_program is None:
#                 logging.warning("Failed to generate a valid program. Retrying...")
#                 attempts += 1
#                 program_recalls += 1
#                 continue
            
#             logging.info(f"Generated Program:\n{generated_program}")
#             output_cpu, error_cpu, output_gpu, error_gpu, explanation = execute_python_code(generated_program)
            
#             if explanation == "Program has syntax errors." or explanation == "Program has a TypeError and needs to be regenerated.":
#                 logging.warning(f"{explanation}. Regenerating...")
#                 attempts += 1
#                 program_recalls += 1
#             elif explanation:
#                 logging.info(f"Program executed successfully but contains bugs:\n{explanation}")
#                 break
#             else:
#                 logging.info("Program executed successfully without bugs.")
#                 break

#     end_time = datetime.now()
#     runtime = end_time - start_time
#     update_statistics(runtime)

#     logging.info("\nFinal Statistics for this run:")
#     with open("statistics.csv", "r") as f:
#         lines = f.readlines()
#         if len(lines) > 1:
#             logging.info(lines[0].strip())  # Log the header
#             logging.info(lines[-1].strip())  # Log the last line (current run statistics)

#     logging.info(f"\nTotal runtime: {runtime}")

# if __name__ == "__main__":
#     main()







import os
import requests
import re
import random
import time
import json
import subprocess
import tempfile
import sys
import csv
from datetime import datetime
from openai import OpenAI
import torch
from concurrent.futures import ThreadPoolExecutor  # For parallel execution

from logging_config import *  # Import the common logging configuration

# Set your API key and base URL for OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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"
)

# Global variables for statistics
total_programs = 0
valid_programs = 0
buggy_programs = 0
api_coverage = {}  # Changed to dictionary to track API usage
program_recalls = 0
start_time = None

def update_statistics(runtime, log_file="results.txt"):
    file_exists = os.path.isfile("statistics.csv")
    
    with open("statistics.csv", "a", newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Total programs", "Valid programs", "Buggy programs", "API coverage", "Program recalls", "Runtime (seconds)"])
        
        row = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            total_programs,
            valid_programs,
            buggy_programs,
            f"{len(api_coverage)}/{len(dir(torch))}",
            program_recalls,
            round(runtime.total_seconds(), 2)
        ]
        
        writer.writerow(row)
        logging.info(f"Statistics updated: {row}")  # Log statistics to file

def save_buggy_program(code, index):
    if not os.path.exists("buggy-programs"):
        os.makedirs("buggy-programs")
    filename = f"buggy_program_{index}.py"
    clean_code = clean_code_snippet(code)
    with open(os.path.join("buggy-programs", filename), "w") as f:
        f.write(clean_code)
    logging.info(f"Buggy program saved as: {filename}")

def clean_code_snippet(code):
    return code.strip()

# Function to read APIs from a .csv file
def get_generated_apis(filename="apis.csv"):
    """
    Load API calls from a CSV file, ignoring the header.

    Args:
    - filename: Name of the CSV file (default: apis.csv)

    Returns:
    - List of tuples where each tuple is (api_name, parameters)
    """
    api_list = []
    try:
        with open(filename, mode='r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # Skip the header row
            for row in csv_reader:
                api_name = row[0].strip()  # First column is the API name
                parameters = [param.strip() for param in row[1].split(',')]  # Split parameters by commas and trim whitespaces
                api_list.append((api_name, parameters))
        logging.info(f"APIs loaded successfully from {filename}")
    except FileNotFoundError:
        logging.error(f"API file {filename} not found.")
    return api_list

def analyze_pytorch_bug(api_name, parameters, variation=None):
    """
    Generate a new test program based on the API and parameters.
    Variation: Add some prompt variation if a retry is required.
    """
    prompt_variation = f"\nTry a simpler example." if variation == "simple" else ""
    formatted_parameters = ', '.join(parameters)
    prompt = f"""
Generate a runnable Python PyTorch program to test the API {api_name} with the following parameters:
{formatted_parameters}. {prompt_variation}

The program should:
- Run without errors.
- Use the parameters and produce realistic test values (e.g., tensor data).
- Show potential issues or differences between CPU and GPU.
- Ensure reproducibility by setting random seeds.

Provide only the complete code, without any additional explanations.
"""
    try:
        response = client.chat.completions.create(
            model="nousresearch/hermes-3-llama-3.1-405b:free",
            messages=[
                {"role": "system", "content": "You are a PyTorch developer tasked with creating example programs for API testing."},
                {"role": "user", "content": prompt}
            ]
        )
        
        if response and response.choices and len(response.choices) > 0 and response.choices[0].message:
            generated_code = response.choices[0].message.content
            generated_code = generated_code.replace("```python", "").replace("```", "")
            generated_code = generated_code.replace('api_name', api_name)
            for i, param in enumerate(parameters):
                generated_code = generated_code.replace(f'param{i+1}', param)
            logging.info(f"Generated program for API {api_name}")
            return clean_code_snippet(generated_code)
        else:
            logging.error("Error: Unexpected response structure from the API.")
            return None
    except Exception as e:
        logging.error(f"Error occurred while generating new program: {str(e)}")
        return None

def execute_python_code(code, api_name, parameters):
    global total_programs, valid_programs, buggy_programs, api_coverage

    clean_code = clean_code_snippet(code)
    seed = random.randint(0, 2**32 - 1)
    seeded_code = f"""
import random
import numpy as np
import torch

random.seed({seed})
np.random.seed({seed})
torch.manual_seed({seed})
if torch.cuda.is_available():
    torch.cuda.manual_seed_all({seed})

{clean_code}
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
        temp_file.write(seeded_code)
        temp_file_path = temp_file.name

    try:
        python_executable = sys.executable
        result_cpu = subprocess.run([python_executable, temp_file_path], capture_output=True, text=True, timeout=10, env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1"))
        output_cpu = result_cpu.stdout
        error_cpu = result_cpu.stderr

        if torch.cuda.is_available():
            result_gpu = subprocess.run([python_executable, temp_file_path], capture_output=True, text=True, timeout=10, env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"))
            output_gpu = result_gpu.stdout
            error_gpu = result_gpu.stderr
        else:
            output_gpu = output_cpu
            error_gpu = error_cpu

        # Enhanced error handling for different error types
        if "SyntaxError" in error_cpu or "SyntaxError" in error_gpu:
            logging.error(f"SyntaxError Detected in Program Execution for API {api_name}\nCPU Error: {error_cpu}\nGPU Error: {error_gpu}")
            return None, "Syntax error detected", None, None, "Program has syntax errors."

        if "TypeError" in error_cpu or "TypeError" in error_gpu:
            logging.error(f"TypeError Detected in Program Execution for API {api_name}\nCPU Error: {error_cpu}\nGPU Error: {error_gpu}")
            return None, "TypeError detected", None, None, "Program has a TypeError and needs to be regenerated."

        if "RuntimeError" in error_cpu or "RuntimeError" in error_gpu:
            logging.error(f"RuntimeError Detected in Program Execution for API {api_name}\nCPU Error: {error_cpu}\nGPU Error: {error_gpu}")
            return None, "RuntimeError detected", None, None, "Program has a RuntimeError."

        # Compare CPU and GPU outputs
        if output_cpu != output_gpu or error_cpu != error_gpu:
            buggy_programs += 1
            save_buggy_program(seeded_code, buggy_programs)
            explanation = f"Program produced different results on CPU vs GPU."
            logging.warning(f"CPU Output: {output_cpu}\nCPU Error: {error_cpu}\nGPU Output: {output_gpu}\nGPU Error: {error_gpu}")
            return output_cpu, error_cpu, output_gpu, error_gpu, explanation

        valid_programs += 1
        # Track API usage
        for line in code.split('\n'):
            if 'torch.' in line:
                api_call = line.split('torch.')[1].split('(')[0]
                if api_call in api_coverage:
                    api_coverage[api_call] += 1
                else:
                    api_coverage[api_call] = 1

        logging.info(f"Program for API {api_name} executed successfully without bugs.")
        return output_cpu, error_cpu, output_gpu, error_gpu, "Program executed successfully without bugs."

    except subprocess.TimeoutExpired:
        explanation = "Execution timed out after 10 seconds"
        logging.error(explanation)
        return "Execution timed out after 10 seconds", "", "", "", explanation
    except Exception as e:
        logging.error(f"Error during program execution for API {api_name}: {e}")
        return None, f"Execution failed: {e}", None, None, "Execution failed."
    finally:
        os.unlink(temp_file_path)

def main():
    global program_recalls, total_programs, valid_programs, buggy_programs, start_time

    logging.info("Starting Automated PyTorch API Tester...")
    start_time = datetime.now()

    apis = get_generated_apis("apis.csv")
    
    logging.info("Analyzing and generating programs for the APIs listed in apis.csv:")

    def analyze_and_execute(api_index, api_name, parameters):
        global total_programs, valid_programs, buggy_programs, program_recalls

        total_programs += 1
        logging.info(f"Program {total_programs} - {api_name} with parameters {parameters}")

        attempts = 0
        max_attempts = 3
        while attempts < max_attempts:
            generated_program = analyze_pytorch_bug(api_name, parameters)
            if generated_program is None:
                logging.warning("Failed to generate a valid program. Retrying...")
                attempts += 1
                program_recalls += 1
                continue

            output_cpu, error_cpu, output_gpu, error_gpu, explanation = execute_python_code(generated_program, api_name, parameters)

            if explanation == "Program has syntax errors." or explanation == "Program has a TypeError and needs to be regenerated.":
                logging.warning(f"{explanation}. Regenerating...")
                attempts += 1
                program_recalls += 1
            elif explanation:
                logging.info(f"Program executed successfully but contains bugs:\n{explanation}")
                break
            else:
                logging.info("Program executed successfully without bugs.")
                break

    # Run the programs in parallel
    with ThreadPoolExecutor(max_workers=5) as executor:
        for api_index, (api_name, parameters) in enumerate(apis):
            executor.submit(analyze_and_execute, api_index, api_name, parameters)

    end_time = datetime.now()
    runtime = end_time - start_time
    update_statistics(runtime)

    logging.info("\nFinal Statistics for this run:")
    with open("statistics.csv", "r") as f:
        lines = f.readlines()
        if len(lines) > 1:
            logging.info(lines[0].strip())  # Log the header
            logging.info(lines[-1].strip())  # Log the last line (current run statistics)

    logging.info(f"\nTotal runtime: {runtime}")

if __name__ == "__main__":
    main()
