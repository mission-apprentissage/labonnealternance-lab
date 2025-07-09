from flask import Flask
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
    return "Hello from your Dockerized Flask app! Environnement: " + env + ""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port)
