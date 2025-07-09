from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
from classifier import Classifier

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()
# Accéder aux variables d'environnement
env = os.getenv('LAB_ENV')
port = os.getenv('LAB_SERVER_PORT')


app = Flask(__name__)
model = Classifier("2025-05-27 offres_ft_rf.joblib")

@app.route("/")
def hello():
    # return "Hello from your Dockerized Flask app! Environnement: " + env + ""
    return "LBA classifier API ready."

@app.route('/score', methods = ['POST'])
def score():
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
    app.run(host="0.0.0.0", port=port)
