from flask import Flask
from dotenv import load_dotenv
import os
# Charger les variables d'environnement depuis le fichier .env
load_dotenv()
# Accéder aux variables d'environnement
public_version = os.getenv('PUBLIC_VERSION')


app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello from your Dockerized Flask app! Version: " + public_version + ""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
