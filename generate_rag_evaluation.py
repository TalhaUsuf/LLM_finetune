import os
from dotenv import load_dotenv, find_dotenv
from groq import Groq
import logging
from rich.logging import RichHandler
from rich.console import Console


load_dotenv(find_dotenv())

console = Console()
script_name = os.path.basename(__file__).replace(".py", "")

# Define the log format
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

log_file_path = "logs"
log_file_name = f"{script_name}.log"


if not os.path.exists(log_file_path):
    os.makedirs(log_file_path)


def get_logger(script_name):
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.DEBUG)

    # Create a file handler
    file_handler = logging.FileHandler(os.path.join(log_file_path, log_file_name))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format))

    # Create a console handler
    console_handler = RichHandler(console=console, level=logging.DEBUG, show_time=True, show_level=True)
    console_handler.setFormatter(logging.Formatter(log_format))

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# get this script's name

logger = get_logger(script_name)


try:
    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
    )
    logger.info("Connected to Groq API", extra={'name': script_name})
except Exception as e:
    logger.error(f"Error connecting to Groq API: {e}", extra={'name': script_name})
    raise e

else:
    chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Explain the importance of fast language models",
        }
    ],
    model="mixtral-8x7b-32768",
)

print(chat_completion.choices[0].message.content)
