import logging
from flask import request, jsonify

logger = logging.getLogger(__name__)


def register_routes(app, get_model):
    """Register inference routes."""

    @app.route('/model/score', methods=['POST'])
    def score():
        if not request.is_json:
            logger.warning("Non-JSON request received on /score")
            return jsonify({'error': 'Request must be JSON'}), 400

        data = request.get_json()
        version = data.get('version')

        model = get_model(version)
        if model is None or not hasattr(model, 'classifier') or model.classifier is None:
            logger.error("No model loaded in memory")
            return jsonify({'error': 'No model loaded in memory. Please load a model first.'}), 503

        texts = {}
        texts['workplace_name'] = data.get('workplace_name')
        texts['workplace_description'] = data.get('workplace_description')
        texts['offer_title'] = data.get('offer_title')
        texts['offer_description'] = data.get('offer_description')
        logger.debug("Received /model/score data: %s", data)

        for key in texts.keys():
            if not isinstance(texts[key], list):
                logger.warning(f"Invalid /model/score payload: '{key}' field is not a list")
                return jsonify({'error': f'"{key}" field must be a list.'}), 400

        result = model.score(texts)
        logger.info("Score computed for texts")
        return jsonify(result), 200

    @app.route('/model/scores', methods=['POST'])
    def scores():
        if not request.is_json:
            logger.warning("Non-JSON request received on /scores")
            return jsonify({'error': 'Request must be JSON'}), 400

        data = request.get_json()
        version = data.get('version')

        model = get_model(version)
        if model is None or not hasattr(model, 'classifier') or model.classifier is None:
            logger.error("No model loaded in memory")
            return jsonify({'error': 'No model loaded in memory. Please load a model first.'}), 503

        items = data.get('items')
        logger.debug("Received /model/scores data: %s", data)

        if not isinstance(items, list):
            logger.warning("Invalid /model/scores payload: 'items' is not a list")
            return jsonify({'error': '"items" must be a list.'}), 400

        required_fields = ['id', 'workplace_name', 'workplace_description', 'offer_title', 'offer_description']
        text_fields = ['workplace_name', 'workplace_description', 'offer_title', 'offer_description']

        # Validate all items first
        for idx, item in enumerate(items):
            if not isinstance(item, dict) or not all(f in item for f in required_fields):
                logger.warning("Invalid item at index %d: missing required fields", idx)
                return jsonify({'error': f'Each item must have {required_fields}.'}), 400
            if not isinstance(item['id'], str):
                logger.warning("Invalid item types at index %d: 'id' is not a str", idx)
                return jsonify({'error': '"id" must be a str.'}), 400

        texts = {
            'workplace_name': [item['workplace_name'] or '' for item in items],
            'workplace_description': [item['workplace_description'] or '' for item in items],
            'offer_title': [item['offer_title'] or '' for item in items],
            'offer_description': [item['offer_description'] or '' for item in items],
        }
        ids = [item['id'] for item in items]
        batch_scores = model.score(texts)
        results = []
        for item_id, score_result in zip(ids, batch_scores):
            results.append({'id': item_id, **score_result})

        logger.info("Batch scored: %d items", len(results))
        return jsonify(results), 200
