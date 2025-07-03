# Utilise l'image officielle Python 3.12 slim
FROM python:3.12-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les dépendances en premier pour profiter du cache Docker
COPY server/requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste de l'application
COPY server/ .

EXPOSE 5000

# Commande de lancement (à adapter selon ton point d'entrée)
CMD ["python", "main.py"]
