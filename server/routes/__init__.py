import logging
from flask import request

logger = logging.getLogger(__name__)


def register_all_routes(app, get_model):
    """
    Register all routes from different modules.

    Args:
        app: Flask application instance
        get_model: Function to get the current model
    """
    # Import route modules
    from . import health, model, inference, training, evaluation

    # Register middleware for request logging
    @app.before_request
    def log_request():
        if request.path != '/favicon.ico':
            logger.info("%s %s", request.method, request.path)

    # Register routes from each module
    health.register_routes(app)
    model.register_routes(app, get_model)
    inference.register_routes(app, get_model)
    training.register_routes(app, get_model)
    evaluation.register_routes(app, get_model)
