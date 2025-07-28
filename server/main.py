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
