import logging

# Set up logging configuration to log messages to a file and the console

# Configure the basic settings for the logging system
logging.basicConfig(
    filename='combined_log.log',  # Specify the file where log messages will be stored
    filemode='a',  # Open the file in append mode ('a') to add new log entries without overwriting existing ones
    level=logging.INFO,  # Set the logging level to INFO; logs messages of level INFO and above (INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s'  # Format of each log message: includes timestamp, log level, and message
)

# Create a StreamHandler to output log messages to the console
console = logging.StreamHandler()  # StreamHandler directs log output to the console (standard output)
console.setLevel(logging.INFO)  # Set the console logging level to INFO (to match the file logger)

# Create a Formatter object to define how the log messages should be displayed
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')  # Format log messages with a timestamp, log level, and message

# Attach the formatter to the console handler to ensure console output is properly formatted
console.setFormatter(formatter)

# Add the console handler to the root logger
# This ensures that log messages are displayed both in the log file and in the console
logging.getLogger('').addHandler(console)
