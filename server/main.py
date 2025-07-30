from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
from classifier import Classifier

load_dotenv()

port = os.getenv('LAB_SERVER_PORT')

app = Flask(__name__)
model = Classifier("models/2025-07-28 offres_ft_svc.pkl")

@app.route("/")
def ready():
    """
    Default route that returns a message indicating the API is ready.

    Returns:
        str: A message indicating that the LBA classifier API is ready.
    """
    return "LBA classifier API ready."

@app.route("/version")
def version():
    """
    Route to get classifier model version.

    Returns:
        Response: A JSON object containing the version of the classifier model.
    """
    return jsonify({'model': model.classifier_name}), 200

@app.route('/score', methods = ['POST'])
def score():
    """
    Route to score a text using the Classifier model.
    Accepts only POST requests with JSON data.

    Returns:
        Response: A JSON object containing the text score or an error if the request is invalid.
    """
    # print("Received request:", request)
    if request.is_json:
        # Get the JSON data
        data = request.get_json() 
        print("Received data:", data)
        text = data['text']
        return jsonify(model.score(text)), 200
    else:
        # If the request did not contain JSON data, return an error
        return jsonify({'error': 'Request must be JSON'}), 400

@app.route('/scores', methods=['POST'])
def scores():
    """
    Route to score multiple texts using the Classifier model.
    Each input must be an object with 'id' and 'text' fields.

    Returns:
        Response: A list of objects with 'id' and the corresponding score.
    """
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400

    data = request.get_json()
    print("Received batch data:", data)

    if 'items' not in data:
        return jsonify({'error': 'Missing "items" field (list of {id, text}).'}), 400

    items = data['items']
    if not isinstance(items, list):
        return jsonify({'error': '"items" must be a list.'}), 400

    results = []
    for item in items:
        if not isinstance(item, dict) or 'id' not in item or 'text' not in item:
            return jsonify({'error': 'Each item must be an object with "id" and "text".'}), 400
        if not isinstance(item['id'], str) or not isinstance(item['text'], str):
            return jsonify({'error': '"id" must be str, and "text" must be a string.'}), 400

        score = model.score(item['text'])
        results.append({'id': item['id'], **score})

    return jsonify(results), 200

if __name__ == "__main__":
    # Start the Flask application on the specified host and port
    app.run(host="0.0.0.0", port=port)


