import logging
from flask import Flask, jsonify
from classifier import Classifier
import os

# Global model variable - will be initialized in worker
model = None
token = os.environ['LBA_HF_TOKEN']

def get_model(version=None):
    """Lazy load model in worker process to avoid CUDA forking issues"""
    global model
    if version is None:
        return model

    if model is None:
        # Initialize classifier
        model = Classifier(version=version, token=token)

        # Reload model
        model.load_model()
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
