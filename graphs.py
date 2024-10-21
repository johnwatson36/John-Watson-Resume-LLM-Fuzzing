# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('Agg')  # Use a non-GUI backend to avoid display issues

# import torch
# from logging_config import *  # Import the common logging configuration

# # Function to check if an API exists in the PyTorch library
# def is_valid_pytorch_api(api_name):
#     try:
#         components = api_name.split('.')
#         module = torch

#         # Dynamically traverse the PyTorch module hierarchy to check for the API
#         for comp in components:
#             module = getattr(module, comp)

#         return callable(module)
#     except AttributeError:
#         logging.warning(f"Invalid PyTorch API: {api_name}")
#         return False

# def validate_pytorch_apis(df):
#     logging.info("Validating PyTorch APIs...")
    
#     # Iterate over the DataFrame rows and validate each API
#     for index, row in df.iterrows():
#         api_name = row['api_name']
#         if is_valid_pytorch_api(api_name):
#             df.at[index, 'is_valid'] = True
#             logging.info(f"API '{api_name}' is valid.")
#         else:
#             df.at[index, 'is_valid'] = False
#             logging.info(f"API '{api_name}' is invalid.")
    
#     # Filter out invalid APIs
#     df_valid = df[df['is_valid'] == True].copy()  # Keep only valid APIs
#     df_invalid = df[df['is_valid'] == False]  # For logging invalid APIs

#     if not df_invalid.empty:
#         logging.info(f"Removing {len(df_invalid)} invalid APIs from the CSV file.")
    
#     # Drop the 'is_valid' column to preserve the original format (api_name, parameters)
#     df_valid = df_valid.drop(columns=['is_valid'])

#     # Save the updated DataFrame to the CSV file, excluding invalid APIs, with the original format
#     df_valid.to_csv('apis.csv', index=False, columns=['api_name', 'parameters'])  # Ensure only 'api_name' and 'parameters' are saved
#     logging.info(f"Updated apis.csv file saved with valid APIs only.")

# def create_graphs():
#     # Load the DataFrame from the apis.csv file
#     try:
#         df = pd.read_csv('apis.csv')
#         logging.info("Loaded apis.csv successfully.")
#         logging.info(f"Data columns: {df.columns}")
#     except FileNotFoundError:
#         logging.error("apis.csv file not found. Please ensure the file is available.")
#         return

#     # Apply transformation to calculate number of parameters
#     df['num_parameters'] = df['parameters'].apply(lambda x: len(x.split(',')))
#     logging.info("Number of parameters calculated for each API.")

#     # Combine API name and parameters to check for full API call uniqueness
#     df['api_full_call'] = df['api_name'] + df['parameters']
#     logging.info("API full call uniqueness calculated.")

#     # Check for unique API calls (True if unique, False if duplicated)
#     df['is_unique_call'] = ~df.duplicated('api_full_call')
#     logging.info("Uniqueness of API calls determined.")

#     # Add a column to track if the API is valid
#     df['is_valid'] = False

#     # Validate the APIs in PyTorch
#     validate_pytorch_apis(df)

#     # Create directory for saving graphs if it doesn't exist
#     output_dir = 'api_graphical_results'
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#         logging.info(f"Created directory: {output_dir}")

#     # Visualization: Number of parameters per API
#     plt.figure(figsize=(10, 6))
#     plt.barh(df['api_name'], df['num_parameters'], color='skyblue')
#     plt.xlabel('Number of Parameters')
#     plt.title('Number of Parameters in Generated API Calls')
#     plt.tight_layout()
#     params_per_api_path = os.path.join(output_dir, 'parameters_per_api.png')
#     plt.savefig(params_per_api_path)
#     logging.info(f"Graph for number of parameters per API created and saved as '{params_per_api_path}'.")

#     # Visualization: Uniqueness of API calls (name + parameters)
#     plt.figure(figsize=(10, 6))
#     plt.bar(df['api_name'], df['is_unique_call'].astype(int), color='lightblue')
#     plt.ylabel('Unique API Call (1=True, 0=False)')
#     plt.xticks(rotation=90)
#     plt.title('Uniqueness of API Calls (API Name + Parameters)')
#     plt.tight_layout()
#     uniqueness_path = os.path.join(output_dir, 'api_call_uniqueness.png')
#     plt.savefig(uniqueness_path)
#     logging.info(f"Graph for API call uniqueness created and saved as '{uniqueness_path}'.")

#     # Visualization: Validity of PyTorch APIs
#     plt.figure(figsize=(10, 6))
#     plt.bar(df['api_name'], df['is_valid'].astype(int), color='green')
#     plt.ylabel('Valid API (1=True, 0=False)')
#     plt.xticks(rotation=90)
#     plt.title('Validity of PyTorch API Calls')
#     plt.tight_layout()
#     validity_path = os.path.join(output_dir, 'validity_of_apis.png')
#     plt.savefig(validity_path)
#     logging.info(f"Graph for API validity created and saved as '{validity_path}'.")

# # To run the create_graphs function
# if __name__ == "__main__":
#     logging.info("Starting PyTorch API validation and graph generation.")
#     create_graphs()
#     logging.info("Graph generation completed.")

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend to avoid display issues

import torch
from logging_config import *  # Import the common logging configuration

# Function to check if an API exists in the PyTorch library
def is_valid_pytorch_api(api_name):
    try:
        components = api_name.split('.')
        module = torch

        # Dynamically traverse the PyTorch module hierarchy to check for the API
        for comp in components:
            module = getattr(module, comp)

        return callable(module)
    except AttributeError:
        logging.warning(f"Invalid PyTorch API: {api_name}")
        return False

# Function to validate APIs and add a validity column
def validate_pytorch_apis(df):
    logging.info("Validating PyTorch APIs...")
    
    # Iterate over the DataFrame rows and validate each API
    for index, row in df.iterrows():
        api_name = row['api_name']
        if is_valid_pytorch_api(api_name):
            df.at[index, 'is_valid'] = True
            logging.info(f"API '{api_name}' is valid.")
        else:
            df.at[index, 'is_valid'] = False
            logging.info(f"API '{api_name}' is invalid.")

# Function to clean the CSV file by removing invalid APIs
def clean_invalid_apis(df):
    # Filter out invalid APIs
    df_valid = df[df['is_valid'] == True].copy()  # Keep only valid APIs
    df_invalid = df[df['is_valid'] == False]  # For logging invalid APIs

    if not df_invalid.empty:
        logging.info(f"Removing {len(df_invalid)} invalid APIs from the CSV file.")
    
    # Drop the 'is_valid' column to preserve the original format (api_name, parameters)
    df_valid = df_valid.drop(columns=['is_valid'])

    # Save the updated DataFrame to the CSV file, excluding invalid APIs, with the original format
    df_valid.to_csv('apis.csv', index=False, columns=['api_name', 'parameters'])  # Ensure only 'api_name' and 'parameters' are saved
    logging.info(f"Updated apis.csv file saved with valid APIs only.")

# Function to create graphs based on original data (before cleaning)
def create_graphs():
    # Load the DataFrame from the apis.csv file
    try:
        df = pd.read_csv('apis.csv')
        logging.info("Loaded apis.csv successfully.")
        logging.info(f"Data columns: {df.columns}")
    except FileNotFoundError:
        logging.error("apis.csv file not found. Please ensure the file is available.")
        return

    # Apply transformation to calculate number of parameters
    df['num_parameters'] = df['parameters'].apply(lambda x: len(x.split(',')))
    logging.info("Number of parameters calculated for each API.")

    # Combine API name and parameters to check for full API call uniqueness
    df['api_full_call'] = df['api_name'] + df['parameters']
    logging.info("API full call uniqueness calculated.")

    # Check for unique API calls (True if unique, False if duplicated)
    df['is_unique_call'] = ~df.duplicated('api_full_call')
    logging.info("Uniqueness of API calls determined.")

    # Add a column to track if the API is valid
    df['is_valid'] = False

    # Validate the APIs in PyTorch
    validate_pytorch_apis(df)

    # Create directory for saving graphs if it doesn't exist
    output_dir = 'api_graphical_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created directory: {output_dir}")

    # Visualization: Number of parameters per API
    plt.figure(figsize=(10, 6))
    plt.barh(df['api_name'], df['num_parameters'], color='skyblue')
    plt.xlabel('Number of Parameters')
    plt.title('Number of Parameters in Generated API Calls')
    plt.tight_layout()
    params_per_api_path = os.path.join(output_dir, 'parameters_per_api.png')
    plt.savefig(params_per_api_path)
    logging.info(f"Graph for number of parameters per API created and saved as '{params_per_api_path}'.")

    # Visualization: Uniqueness of API calls (name + parameters)
    plt.figure(figsize=(10, 6))
    plt.bar(df['api_name'], df['is_unique_call'].astype(int), color='lightblue')
    plt.ylabel('Unique API Call (1=True, 0=False)')
    plt.xticks(rotation=90)
    plt.title('Uniqueness of API Calls (API Name + Parameters)')
    plt.tight_layout()
    uniqueness_path = os.path.join(output_dir, 'api_call_uniqueness.png')
    plt.savefig(uniqueness_path)
    logging.info(f"Graph for API call uniqueness created and saved as '{uniqueness_path}'.")

    # Visualization: Validity of PyTorch APIs
    plt.figure(figsize=(10, 6))
    plt.bar(df['api_name'], df['is_valid'].astype(int), color='green')
    plt.ylabel('Valid API (1=True, 0=False)')
    plt.xticks(rotation=90)
    plt.title('Validity of PyTorch API Calls')
    plt.tight_layout()
    validity_path = os.path.join(output_dir, 'validity_of_apis.png')
    plt.savefig(validity_path)
    logging.info(f"Graph for API validity created and saved as '{validity_path}'.")

    # Clean invalid APIs after generating the graphs
    clean_invalid_apis(df)

# To run the create_graphs function
if __name__ == "__main__":
    logging.info("Starting PyTorch API validation and graph generation.")
    create_graphs()
    logging.info("Graph generation and API cleaning completed.")
