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
PUBLIC_VERSION = os.getenv('PUBLIC_VERSION', 'unknown')

# Training token (outbound: used to authenticate to the LBA classification API)
LBA_API_TOKEN = os.getenv('LBA_API_TOKEN')

# Admin token (inbound: required to call sensitive routes like /model/train)
LAB_ADMIN_TOKEN = os.getenv('LAB_ADMIN_TOKEN')

# Allowlist of hosts the training endpoint is allowed to fetch from (SSRF guard).
# Comma-separated override via TRAIN_ALLOWED_HOSTS; defaults to the official LBA API host.
TRAIN_ALLOWED_HOSTS = {
    host.strip()
    for host in os.getenv(
        'TRAIN_ALLOWED_HOSTS',
        'labonnealternance.apprentissage.beta.gouv.fr',
    ).split(',')
    if host.strip()
}
