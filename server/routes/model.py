import logging
from flask import request, jsonify

logger = logging.getLogger(__name__)


def register_routes(app, get_model):
    """Register model management routes."""

    @app.route("/model/load", methods=['GET'])
    def load_model():
        version = request.args.get('version')
        if not version:
            log = "'version' argument missing."
            logger.warning(log)
            return jsonify({'error': log}), 400

        logger.info("Received /model/load: %s", version)
        model = get_model(version=version)
        logger.info("Model version ready: %s", model.version)
        return jsonify({'model': model.version}), 200

    @app.route("/model/version", methods=['GET'])
    def model_version():
        model = get_model()
        version = model.version if model else None
        logger.info("Model version loaded: %s", version)
        return jsonify({'model': version}), 200
