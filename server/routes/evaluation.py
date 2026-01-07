import logging
from flask import request, jsonify

logger = logging.getLogger(__name__)


def register_routes(app, get_model):
    """Register evaluation routes."""

    @app.route('/model/evaluate', methods=['POST'])
    def evaluate():
        if not request.is_json:
            logger.warning("Non-JSON request received on /evaluate")
            return jsonify({'error': 'Request must be JSON'}), 400

        data = request.get_json()
        versions = data.get('versions')
        texts = data.get('texts')
        labels = data.get('labels')

        logger.debug("Received /model/evaluate data: %s", data)

        evaluation = {}
        for version in versions:
            evaluation[version] = []
            model = get_model(version)

            if not model.classifier:
                error = f"Model '{version}' do not exist."
                logger.error(error)
                return jsonify({'error': error}), 400

            evaluation[version] = model.evaluate(texts=texts, labels=labels)
            logger.info(f"Model '{version}' scores: {evaluation[version]}")
        return jsonify(evaluation), 200
