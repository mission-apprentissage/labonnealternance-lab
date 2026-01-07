import logging
from flask import request, jsonify

logger = logging.getLogger(__name__)


def register_routes(app, get_model):
    """Register training routes."""

    @app.route('/model/train/local', methods=['POST'])
    def train_model_local():
        if not request.is_json:
            logger.warning("Non-JSON request received on /model/train")
            return jsonify({'error': 'Request must be JSON'}), 400

        data = request.get_json()
        logger.debug("Received /model/train/local data: %s", data)

        version = data.get('version')
        ids = data.get('ids')
        texts = data.get('texts')
        labels = data.get('labels')

        if len(ids) < 12:
            logger.warning("Not enough items for training")
            return jsonify({'error': 'Please provide at least 12 items for dataset training'}), 400

        # Load model
        model = get_model(version)

        # Create and save dataset
        dataset = model.create_dataset_local(version, ids, texts, labels)
        dataset_url = model.save_dataset()
        logger.info(f"Saved dataset '{version}' to '{dataset_url}'")

        # Train and save model
        classifier, train_score, test_score = model.train_model()
        model_url = model.save_model()
        logger.info(f"Saved model '{version}' to '{model_url}'")
        result = {
            'version': version,
            'train_score': round(train_score, 4),
            'test_score': round(test_score, 4),
            'dataset_url': dataset_url,
            'model_url': model_url
        }
        return jsonify(result), 200

    @app.route('/model/train/online', methods=['POST'])
    def train_model_online():
        if not request.is_json:
            logger.warning("Non-JSON request received on /model/train")
            return jsonify({'error': 'Request must be JSON'}), 400

        data = request.get_json()
        logger.debug("Received /model/train/online data: %s", data)
        version = data.get('version')
        endpoint = data.get('endpoint')

        # Load model
        model = get_model(version)

        # Create and save dataset
        dataset = model.create_dataset_online(version, endpoint)
        if len(dataset) > 0:
            dataset_url = model.save_dataset()
            logger.info(f"Saved dataset '{version}' to '{dataset_url}'")

            # Train and save model
            classifier, train_score, test_score = model.train_model()
            model_url = model.save_model()
            logger.info(f"Saved model '{version}' to '{model_url}'")
            result = {
                'version': version,
                'train_score': round(train_score, 4),
                'test_score': round(test_score, 4),
                'dataset_url': dataset_url,
                'model_url': model_url
            }
            return jsonify(result), 200

        else:
            logger.warning("Unvalid dataset")
            return jsonify({'error': 'No valid data available on this endpoint'}), 400
