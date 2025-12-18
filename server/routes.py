import logging
from flask import request, jsonify

logger = logging.getLogger(__name__)

def register_routes(app, get_model):
    @app.route("/")
    def api_ready():
        logger.info("Healthcheck received on /")
        return jsonify({'status': "LBA classifier API ready."})
    
    @app.route("/model/load", methods=['GET'])
    def load_model():
        version = request.args.get('version')
        if not version:
            log = "'version' argument missing."
            logger.warning(log)
            return jsonify({'error': log}), 400

        logger.debug("Received /model/load: %s", version)
        model = get_model(version=version)
        logger.info("Model version ready: %s", model.version)
        return jsonify({'model': model.version}), 200

    @app.route("/model/version", methods=['GET'])
    def model_version():
        model = get_model()
        version = model.version if model else None
        logger.info("Model version loaded: %s", version)
        return jsonify({'model': version}), 200

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

        if len(ids)<12:
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
        result = {'version': version, 
                  'train_score': round(train_score,4), 
                  'test_score': round(test_score,4), 
                  'dataset_url': dataset_url, 
                  'model_url': model_url}
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
        if len(dataset) > 0 :
            dataset_url = model.save_dataset()
            logger.info(f"Saved dataset '{version}' to '{dataset_url}'")

            # Train and save model
            classifier, train_score, test_score = model.train_model()
            model_url = model.save_model()
            logger.info(f"Saved model '{version}' to '{model_url}'")
            result = {'version': version, 
                    'train_score': round(train_score,4), 
                    'test_score': round(test_score,4), 
                    'dataset_url': dataset_url, 
                    'model_url': model_url}
            return jsonify(result), 200
            
        else:
            logger.warning("Unvalid dataset")
            return jsonify({'error': 'No valid data available on this endpoint'}), 400

    @app.route('/model/score', methods=['POST'])
    def score():
        if not request.is_json:
            logger.warning("Non-JSON request received on /score")
            return jsonify({'error': 'Request must be JSON'}), 400

        data = request.get_json()
        version = data.get('version')
        text = data.get('text')
        logger.debug("Received /model/score data: %s", data)

        if not isinstance(text, str):
            logger.warning("Invalid /model/score payload: 'text' is not a string")
            return jsonify({'error': '"text" must be a string.'}), 400

        model = get_model(version)
        result = model.score(text)
        logger.info("Score computed for single text")
        return jsonify(result), 200

    @app.route('/model/scores', methods=['POST'])
    def scores():
        if not request.is_json:
            logger.warning("Non-JSON request received on /scores")
            return jsonify({'error': 'Request must be JSON'}), 400

        data = request.get_json()
        version = data.get('version')
        items = data.get('items')
        logger.debug("Received /model/scores data: %s", data)

        if not isinstance(items, list):
            logger.warning("Invalid /model/scores payload: 'items' is not a list")
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
        model = get_model(version)
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

