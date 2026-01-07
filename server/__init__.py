import logging
from flask import Flask, jsonify
from model_manager import load_latest_model, get_model

logger = logging.getLogger(__name__)


def create_app():
    """
    Create and configure the Flask application.

    Returns:
        Flask: Configured Flask application instance.
    """
    app = Flask(__name__)

    # Configure logging for Gunicorn compatibility
    if __name__ != '__main__':
        gunicorn_logger = logging.getLogger('gunicorn.error')
        app.logger.handlers = gunicorn_logger.handlers
        app.logger.setLevel(gunicorn_logger.level)

    # Load latest model at startup
    load_latest_model()

    # Register all routes
    from routes import register_all_routes
    register_all_routes(app, get_model)

    # Global exception handler
    @app.errorhandler(Exception)
    def handle_exception(e):
        app.logger.error("Unhandled exception: %s", e, exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500

    return app
