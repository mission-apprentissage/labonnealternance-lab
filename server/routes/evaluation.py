import json
import logging
import os
from flask import request, jsonify
import pandas as pd

logger = logging.getLogger(__name__)

_dataset_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'validation_dataset.json')
"""
with open(_dataset_path) as f:
    _dataset = json.load(f)
"""
_dataset = pd.read_json(_dataset_path, lines=True)

def register_routes(app, get_model):
    """Register evaluation routes."""

    @app.route('/model/evaluate', methods=['POST'])
    def evaluate():
        if not request.is_json:
            logger.warning("Non-JSON request received on /evaluate")
            return jsonify({'error': 'Request must be JSON'}), 400

        data = request.get_json()
        versions = data.get('versions')

        if not versions or len(versions) < 2:
            logger.warning("Not enough versions for evaluation")
            return jsonify({'error': 'Please provide at least 2 versions to evaluate'}), 400

        logger.debug("Received /model/evaluate data: %s", data)

        texts = _dataset[['workplace_name', 'workplace_description', 'offer_title', 'offer_description']].to_dict(orient='records')
        labels = _dataset['label'].to_list()

        evaluation = {}
        for version in versions:
            model = get_model(version)

            if not model.classifier:
                error = f"Model '{version}' do not exist."
                logger.error(error)
                return jsonify({'error': error}), 400

            evaluation[version] = model.evaluate(texts=texts, labels=labels)
            logger.info(f"Model '{version}' scores: {evaluation[version]}")
        return jsonify(evaluation), 200
