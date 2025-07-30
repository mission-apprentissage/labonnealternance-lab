import logging
from flask import Flask, jsonify
from classifier import Classifier

model = Classifier("models/2025-07-28 offres_ft_svc.pkl")

def create_app():
    app = Flask(__name__)

    # Enregistre les routes
    from routes import register_routes
    register_routes(app, model)

    # Gestion globale des exceptions
    @app.errorhandler(Exception)
    def handle_exception(e):
        logger = logging.getLogger(__name__)
        logger.error("Unhandled exception: %s", e, exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500

    return app
