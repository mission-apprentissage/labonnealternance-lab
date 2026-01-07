import logging
from flask import Flask, jsonify
from classifier import Classifier
import os
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()

# Global model variable - will be initialized in worker
model = None
HF_TOKEN = os.getenv('HF_TOKEN')
logger = logging.getLogger(__name__)

def get_latest_model_version():
    """
    Retrieve the latest model version available on HuggingFace.

    Returns:
        str: The version name of the latest model, or None if no models found.
    """
    try:
        api = HfApi()
        org_name = "la-bonne-alternance"

        # List all models from the organization
        models = api.list_models(author=org_name, token=HF_TOKEN)

        # Extract version names from model IDs (format: la-bonne-alternance/YYYY-MM-DD)
        versions = []
        for model_info in models:
            model_id = model_info.modelId
            if model_id.startswith(f"{org_name}/"):
                version = model_id.replace(f"{org_name}/", "")
                versions.append(version)

        if not versions:
            logger.warning(f"No models found for organization '{org_name}'")
            return None

        # Sort versions by date (assuming format YYYY-MM-DD)
        versions.sort(reverse=True)
        latest_version = versions[0]

        logger.info(f"Latest model version found: {latest_version}")
        return latest_version

    except Exception as e:
        logger.error(f"Error fetching latest model version: {e}")
        return None

def get_model(version=None):
    """Lazy load model in worker process to avoid CUDA forking issues"""
    global model
    # Default retrieve current model
    if version is None:
        return model

    # Retrieve current model or load new model
    if (model is None) or (model.version != version):
        # Initialize classifier
        model = Classifier(version=version, token=HF_TOKEN)

        # Load existing model
        try:
            model.load_model()
            logger.info(f"Reload existing model {version}")
        except:
            logger.info(f"Create new model {version}")
            pass
    return model

def load_latest_model():
    """
    Load the latest available model from HuggingFace at server startup.
    Raises an exception if no model can be loaded.
    """
    global model

    logger.info("Loading latest model at startup...")
    latest_version = get_latest_model_version()

    if not latest_version:
        error_msg = "No model version found on HuggingFace. Cannot start server without a model."
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    try:
        model = get_model(version=latest_version)
        if model is None or not hasattr(model, 'classifier') or model.classifier is None:
            error_msg = f"Model '{latest_version}' loaded but classifier is not available. Cannot start server."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        logger.info(f"Successfully loaded latest model: {latest_version}")
    except RuntimeError:
        raise
    except Exception as e:
        error_msg = f"Failed to load latest model '{latest_version}': {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e

def create_app():
    app = Flask(__name__)

    # Configure logging for Gunicorn compatibility
    if __name__ != '__main__':
        gunicorn_logger = logging.getLogger('gunicorn.error')
        app.logger.handlers = gunicorn_logger.handlers
        app.logger.setLevel(gunicorn_logger.level)

    # Load latest model at startup
    load_latest_model()

    # Enregistre les routes
    from routes import register_routes
    register_routes(app, get_model)

    # Gestion globale des exceptions
    @app.errorhandler(Exception)
    def handle_exception(e):
        app.logger.error("Unhandled exception: %s", e, exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500

    return app
