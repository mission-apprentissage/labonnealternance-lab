import logging
from flask import request, jsonify

logger = logging.getLogger(__name__)

def register_routes(app, get_model):
    @app.route("/")
    def ready():
        logger.info("Healthcheck received on /")
        return "LBA classifier API ready."

    @app.route("/version")
    def version():
        model = get_model()
        version = model.classifier_name
        logger.info("Model version requested: %s", version)
        return jsonify({'model': version}), 200

    @app.route('/score', methods=['POST'])
    def score():
        if not request.is_json:
            logger.warning("Non-JSON request received on /score")
            return jsonify({'error': 'Request must be JSON'}), 400

        data = request.get_json()
        text = data.get('text')
        logger.debug("Received /score data: %s", data)

        if not isinstance(text, str):
            logger.warning("Invalid /score payload: 'text' is not a string")
            return jsonify({'error': '"text" must be a string.'}), 400

        model = get_model()
        result = model.score(text)
        logger.info("Score computed for single text")
        return jsonify(result), 200

    @app.route('/scores', methods=['POST'])
    def scores():
        if not request.is_json:
            logger.warning("Non-JSON request received on /scores")
            return jsonify({'error': 'Request must be JSON'}), 400

        data = request.get_json()
        items = data.get('items')
        logger.debug("Received /scores data: %s", data)

        if not isinstance(items, list):
            logger.warning("Invalid /scores payload: 'items' is not a list")
            return jsonify({'error': '"items" must be a list.'}), 400

        # Validate all items first
        for idx, item in enumerate(items):
            if not isinstance(item, dict) or 'id' not in item or 'text' not in item:
                logger.warning("Invalid item at index %d: missing 'id' or 'text'", idx)
                return jsonify({'error': 'Each item must have "id" and "text".'}), 400
            if not isinstance(item['id'], str) or not isinstance(item['text'], str):
                logger.warning("Invalid item types at index %d: id=%s, text=%s", idx, type(item['id']), type(item['text']))
                return jsonify({'error': '"id" must be str and "text" must be str.'}), 400

        # GPU is now available! Use batch processing for optimal performance
        model = get_model()
        texts = [item['text'] for item in items]
        ids = [item['id'] for item in items]
        batch_scores = model.score_batch(texts)
        results = []
        for item_id, score_result in zip(ids, batch_scores):
            results.append({'id': item_id, **score_result})
        
        # Fallback to individual processing if needed:
        # results = []
        # for item in items:
        #     score = model.score(item['text'])
        #     results.append({'id': item['id'], **score})

        logger.info("Batch scored: %d items", len(results))
        return jsonify(results), 200
