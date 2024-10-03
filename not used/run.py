import torch
import sys
from io import StringIO
from typing import List, Any
import csv

def run_pytorch_code_with_params(code: str, params_list: List[List[str]]) -> List[dict]:
    """
    Run PyTorch code with different sets of parameters.

    Args:
    - code (str): The code to run.
    - params_list (List[List[str]]): A list of parameter sets to use in the code.

    Returns:
    - List[dict]: Results of the code execution for each parameter set.
    """
    results = []

    # Create a namespace for executing the code
    namespace = {'torch': torch}

    # Compile the code
    try:
        compiled_code = compile(code, '<string>', 'exec')
    except SyntaxError as e:
        return [{'params': {}, 'result': None, 'output': None, 'error': f'Syntax error in code: {str(e)}'}]

    for params in params_list:
        # Prepare the parameter dictionary
        param_dict = {f'param{i+1}': eval(param, namespace) for i, param in enumerate(params)}
        
        # Add param_dict to the namespace
        namespace.update(param_dict)

        # Redirect stdout to capture print statements
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            # Execute the code with the current parameters
            exec(compiled_code, namespace)
            
            # Compare cpu_output and gpu_output
            cpu_output = namespace.get('cpu_output')
            gpu_output = namespace.get('gpu_output')
            
            if cpu_output is not None and gpu_output is not None:
                comparison = torch.allclose(cpu_output, gpu_output.cpu(), rtol=1e-5, atol=1e-8)
                result = f"CPU and GPU outputs are {'equal' if comparison else 'not equal'}"
            else:
                result = "CPU or GPU output not found in the code"

            # Capture any printed output
            printed_output = sys.stdout.getvalue()

            results.append({
                'params': param_dict,
                'result': result,
                'output': printed_output.strip(),
                'error': None
            })
        except Exception as e:
            results.append({
                'params': param_dict,
                'result': None,
                'output': None,
                'error': str(e)
            })
        finally:
            # Restore stdout
            sys.stdout = old_stdout

    return results

def read_apis_from_csv(filename: str) -> List[dict]:
    """
    Read API information from a CSV file.

    Args:
    - filename (str): The path to the CSV file.

    Returns:
    - List[dict]: A list of dictionaries containing API details.
    """
    apis = []
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            api_info = {
                'api_name': row['api_name'],
                'parameters': eval(row['parameters']),  # Convert string to list
                'parameter_descriptions': eval(row['parameter_descriptions']),  # Convert string to list
                'example_usage': row['example_usage']
            }
            apis.append(api_info)
    return apis

if __name__ == "__main__":
    # Read the APIs from the CSV file
    api_list = read_apis_from_csv('apis.csv')

    # Run each API example with fuzzed parameters
    for api in api_list:
        example_usage = api['example_usage']
        parameters = api['parameters']
        
        # Generate sample fuzz test parameters (e.g., from parameters.py)
        param_sets = [[f"torch.tensor([{i}])" for i in range(1, len(parameters) + 1)]]  # Example fuzz parameters

        print(f"Running API: {api['api_name']}")
        results = run_pytorch_code_with_params(example_usage, param_sets)
        
        # Print results
        for result in results:
            print(f"Parameters: {result['params']}, Result: {result['result']}, Error: {result['error']}")
