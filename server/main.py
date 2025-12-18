import os
from dotenv import load_dotenv
import logging

load_dotenv()

from __init__ import create_app  # importe create_app du __init__.py local

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,  # ou DEBUG si tu veux
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

port = int(os.getenv('LAB_SERVER_PORT', 8000))
setup_logging()
app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port)
