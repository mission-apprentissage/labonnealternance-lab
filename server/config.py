import os
from dotenv import load_dotenv

load_dotenv()

# HuggingFace configuration
HF_TOKEN = os.getenv('HF_TOKEN')
ORG_NAME = "la-bonne-alternance"

# Server configuration
SERVER_PORT = int(os.getenv('LAB_SERVER_PORT', 5000))
