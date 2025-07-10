from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
from classifier import Classifier

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Accéder aux variables d'environnement
# env = os.getenv('LAB_ENV')
hf_token = os.getenv('LAB_HF_TOKEN')
port = os.getenv('LAB_SERVER_PORT')

app = Flask(__name__)
model = Classifier(repo_id="LaBonneAlternance/offres-classifier", hf_token=hf_token)

@app.route("/")
def ready():
    """
    Default route that returns a message indicating the API is ready.

    Returns:
        str: A message indicating that the LBA classifier API is ready.
    """
    return "LBA classifier API ready."

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

if __name__ == "__main__":
    # Start the Flask application on the specified host and port
    app.run(host="0.0.0.0", port=port)
