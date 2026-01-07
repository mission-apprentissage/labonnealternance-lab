import logging
from flask import jsonify

logger = logging.getLogger(__name__)


def register_routes(app):
    """Register health check routes."""

    @app.route("/favicon.ico")
    def favicon():
        return '', 204

    @app.route("/")
    def api_ready():
        logger.info("Healthcheck received on /")
        return jsonify({'status': "LBA classifier API ready."})
