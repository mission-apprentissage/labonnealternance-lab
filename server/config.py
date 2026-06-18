import os
from dotenv import load_dotenv

load_dotenv()

# HuggingFace configuration
HF_TOKEN = os.getenv('HF_TOKEN')
ORG_NAME = "la-bonne-alternance"

# Model configuration
MODEL_VERSION = "publish-2026-03-31"
# LANG_MODEL = "almanach/camembertav2-base"
LANG_MODEL = "BAAI/bge-m3"

# Server configuration
SERVER_PORT = int(os.getenv('LAB_SERVER_PORT', 8000))
PUBLIC_VERSION = os.getenv('PUBLIC_VERSION', 'unknown')

# Training token
LBA_API_TOKEN = os.getenv('LBA_API_TOKEN')
