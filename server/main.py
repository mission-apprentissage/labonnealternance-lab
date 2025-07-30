from dotenv import load_dotenv
import os
import logging
from app import create_app

load_dotenv()

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,  # Change à DEBUG si besoin
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=[logging.StreamHandler()]  # stdout pour Docker
    )


port = int(os.getenv('LAB_SERVER_PORT', 5000))
setup_logging()
app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port)
