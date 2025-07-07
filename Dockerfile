# Utilise l'image officielle Python 3.13 slim
FROM python:3.13-slim AS server

# Définir le répertoire de travail
WORKDIR /app

# Copier les dépendances en premier pour profiter du cache Docker
COPY server/requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste de l'application
COPY server/ .

EXPOSE 8000

# Commande de lancement (à adapter selon ton point d'entrée)
CMD ["python", "main.py"]
