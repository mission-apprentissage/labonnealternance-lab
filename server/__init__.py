import logging
from flask import Flask, jsonify
from classifier import Classifier
import os

# Global model variable - will be initialized in worker
model = None
HF_TOKEN = os.environ['HF_TOKEN']
logger = logging.getLogger(__name__)

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

def create_app():
    app = Flask(__name__)
    
    # Configure logging for Gunicorn compatibility
    if __name__ != '__main__':
        gunicorn_logger = logging.getLogger('gunicorn.error')
        app.logger.handlers = gunicorn_logger.handlers
        app.logger.setLevel(gunicorn_logger.level)

    # Enregistre les routes
    from routes import register_routes
    register_routes(app, get_model)

    # Gestion globale des exceptions
    @app.errorhandler(Exception)
    def handle_exception(e):
        app.logger.error("Unhandled exception: %s", e, exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500

    return app
