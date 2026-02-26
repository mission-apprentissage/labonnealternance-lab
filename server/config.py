import os
from dotenv import load_dotenv

load_dotenv()

# HuggingFace configuration
HF_TOKEN = os.getenv('HF_TOKEN')
ORG_NAME = "la-bonne-alternance"

# Model configuration
MODEL_VERSION = "2026-02-20"
LANG_MODEL = "almanach/camembertav2-base"

# Server configuration
SERVER_PORT = int(os.getenv('LAB_SERVER_PORT', 8000))
