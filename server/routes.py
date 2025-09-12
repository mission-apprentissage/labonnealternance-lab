import logging
from flask import request, jsonify

logger = logging.getLogger(__name__)

def register_routes(app, get_model):
    @app.route("/")
    def ready():
        logger.info("Healthcheck received on /")
        return "LBA classifier API ready."
    
    @app.route("/init", methods=['POST'])
    def init():
        if not request.is_json:
            logger.warning("Non-JSON request received on /init")
            return jsonify({'error': 'Request must be JSON'}), 400

        data = request.get_json()
        logger.debug("Received /init data: %s", data)

        version = data.get('version')
        model = get_model(version=version)
        version = model.version
        logger.info("Model version ready: %s", version)
        return jsonify({'model': version}), 200

    @app.route("/version")
    def version():
        model = get_model()
        version = model.version
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

    @app.route('/dataset/create', methods=['POST'])
    def create_dataset():
        if not request.is_json:
            logger.warning("Non-JSON request received on /score")
            return jsonify({'error': 'Request must be JSON'}), 400

        data = request.get_json()
        logger.debug("Received /dataset/create data: %s", data)

        items = data.get('items')
        if not isinstance(items, list):
            logger.warning("Invalid /scores payload: 'items' is not a list")
            return jsonify({'error': '"items" must be a list.'}), 400

        ids = [item['id'] for item in items]
        texts = [item['text'] for item in items]
        labels = [item['label'] for item in items]

        model = get_model()
        dataset = model.create_dataset(ids, texts, labels)
        result = {'dataset': model.version, 'shape': dataset.shape}
        logger.info(f"Dataset '{result.dataset}' created: {result.shape}")
        return jsonify(result), 200

    @app.route('/dataset/save')
    def save_dataset():
        model = get_model()
        url = model.save_dataset()
        result = {'dataset': model.version, 'url': url}
        logger.info(f"Dataset '{result.dataset}' saved on: {result.url}")
        return jsonify(result), 200

    @app.route('/dataset/load')
    def load_dataset():
        model = get_model()
        dataset = model.load_dataset()
        result = {'dataset': model.version, 'shape': dataset.shape}
        logger.info(f"Dataset '{result.dataset}' loaded: {result.shape}")
        return jsonify(result), 200

    @app.route('/model/train')
    def train_model():
        model = get_model()
        classifier, train_score, test_score = model.train_model()
        result = {'model': model.version, 'train_score': train_score, 'test_score': test_score}
        logger.info(f"Model '{result.model}' trained: train={result.train_score} / test={result.test_score}")
        return jsonify(result), 200
    
    @app.route('/model/save')
    def save_model():
        model = get_model()
        url = model.save_model()
        result = {'model': model.version, 'url': url}
        logger.info(f"Model '{result.model}' saved on: {result.url}")
        return jsonify(result), 200
    
    @app.route('/model/load')
    def load_model():
        model = get_model()
        model.load_model()
        result = {'model': model.version, 'status': 'loaded'}
        logger.info(f"Model '{result.model}' {result.status}.")
        return jsonify(result), 200


