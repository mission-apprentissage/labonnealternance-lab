![](https://avatars1.githubusercontent.com/u/63645182?s=200&v=4)

# La bonne alternance - Laboratoire

## Fiche Produit

Service de classification de textes pour offres d'emploi utilisant le machine learning. Le service classifie les textes d'offres d'emploi en trois catégories : "cfa", "entreprise", et "entreprise_cfa".

## Architecture

L'application suit une structure modulaire Flask :

```
server/
├── __init__.py              # Factory Flask et initialisation
├── main.py                  # Point d'entrée de l'application
├── config.py                # Configuration centralisée
├── model_manager.py         # Gestion du modèle ML
├── classifier.py            # Classifier ML avec CamemBERT
└── routes/                  # Routes organisées par fonctionnalité
    ├── __init__.py         # Enregistrement des routes
    ├── health.py           # Health checks
    ├── model.py            # Gestion du modèle
    ├── inference.py        # Inférence (score/scores)
    ├── training.py         # Entraînement
    └── evaluation.py       # Évaluation
```

**Technologies** :

- **Language Model** : `almanach/camembertav2-base` pour les embeddings
- **Classification** : Scikit-learn (SVC) chargé depuis HuggingFace
- **Catégories** : cfa, entreprise, entreprise_cfa

## Documentation

## 1. Test application

### Quick Start (npm-style commands)

```shell
make help          # Show available commands
make install       # Install Python dependencies locally
make dev           # Run development server locally with hot-reload
make dev-up        # Start with Docker Compose (development with hot-reload)
make down          # Stop all services
make test          # Test API endpoints
```

### Install requirements

```shell
make install
```

### Environment Configuration

Initialize your local environment configuration:

```shell
$ .bin/mna-lab init:env
```

This command will create/update the `.env` file in the `server/` directory with the required configuration:

```shell
HF_TOKEN=your_huggingface_token_here
LAB_SERVER_PORT=8000  # Optional, default is 8000
```

**Note**: Make sure to update the `HF_TOKEN` value with your actual HuggingFace token if needed.

### Running development server

**Option 1 : Local (recommandé pour développement)**

```shell
$ make dev
```

Lance le serveur en local avec hot-reload activé. Les modifications de code rechargent automatiquement le serveur.

**Option 2 : Docker**

```shell
$ make dev-up
```

Lance le serveur dans Docker avec hot-reload via volume monté.

**Note importante** : Le serveur charge automatiquement le dernier modèle disponible sur HuggingFace au démarrage. Si aucun modèle n'est disponible, le serveur ne démarre pas et affiche un message d'erreur clair dans les logs.

### Test endpoints

#### Check API status

```shell
$ curl http://127.0.0.1:8000/

{"status":"LBA classifier API ready."}
```

#### Check model version (auto-loaded at startup)

```shell
$ curl http://127.0.0.1:8000/model/version

{"model":"2025-12-18"}
```

#### Load specific model version

```shell
$ curl http://127.0.0.1:8000/model/load?version=2025-08-06

{"model":"2025-08-06"}
```

#### Train model version

```shell
$ curl http://127.0.0.1:8000/model/train -X POST -H 'Content-Type: application/json' -d '{"version": "2025-10-24", "endpoint":"https://labonnealternance.apprentissage.beta.gouv.fr/api/classification"}'

{"dataset_url":"https://huggingface.co/datasets/la-bonne-alternance/2025-10-24","model_url":"https://huggingface.co/la-bonne-alternance/2025-10-24","test_score":0.3333,"train_score":0.8888,"version":"2025-10-24"}
```

#### Score texts (without version - uses auto-loaded model)
```shell
$ curl http://127.0.0.1:8000/model/score \
  -X POST \
  -H 'Content-Type: application/json' \
  -d '{
    "workplace_name": ["Adecco"],
    "workplace_description": ["L'\''agence Adecco de Nantes recrute et forme des ajusteurs monteurs H/F pour son client spécialisé dans le domaine de l'\''aéronautique Airbus Atlantic situé à Bouguenais (44020). Chez Airbus Atlantic, tous les Airbus naissent à Nantes ! Le site se spécialise dans la fabrication de caissons centraux de voilure et d'\''autres composants essentiels, utilisant des matériaux composites et de l'\''aluminium de grandes dimensions. Avec le Technocentre et le ZEDC, nous innovons pour l'\''aviation de demain. Airbus Atlantic, partenaire aéronautique global, est une entreprise renommée dans le secteur de l'\''aéronautique. Elle offre un environnement de travail dynamique et stimulant, avec de nombreuses opportunités d'\''évolution professionnelle. En rejoignant Airbus Atlantic, vous aurez la chance de travailler sur des projets passionnants et de contribuer au développement de technologies de pointe dans le domaine de l'\''aéronautique."],
    "offer_title": ["Form. Ajusteur Monteur Aéronautique H/F"],
    "offer_description": ["Détails de la formation : La formation se déroulera à Bouguenais pendant 6 mois. \nThéorie en centre de formation 3 mois : La préparation de l'\''assemblage de structures aéronefs : \nVérifier l'\''approvisionnement du matériel, outils, composants nécessaires aux opérations de montage d'\''éléments mécaniques. Ajuster les portées d'\''un élément sur une structure suivant un ou plusieurs plans [...]"]
  }'

[
  {
    "label": "publish",
    "model": "2026-02-20",
    "scores": {
      "publish": 1.0,
      "unpublish": 0.0
    }
  }
]
```

#### Score texts (with specific version)

```shell
$ curl http://127.0.0.1:8000/model/score \
  -X POST \
  -H 'Content-Type: application/json' \
  -d '{
    "version": "2026-02-20",
    "workplace_name": ["Adecco"],
    "workplace_description": ["L'\''agence Adecco de Nantes recrute et forme des ajusteurs monteurs H/F pour son client spécialisé dans le domaine de l'\''aéronautique Airbus Atlantic situé à Bouguenais (44020). Chez Airbus Atlantic, tous les Airbus naissent à Nantes ! Le site se spécialise dans la fabrication de caissons centraux de voilure et d'\''autres composants essentiels, utilisant des matériaux composites et de l'\''aluminium de grandes dimensions. Avec le Technocentre et le ZEDC, nous innovons pour l'\''aviation de demain. Airbus Atlantic, partenaire aéronautique global, est une entreprise renommée dans le secteur de l'\''aéronautique. Elle offre un environnement de travail dynamique et stimulant, avec de nombreuses opportunités d'\''évolution professionnelle. En rejoignant Airbus Atlantic, vous aurez la chance de travailler sur des projets passionnants et de contribuer au développement de technologies de pointe dans le domaine de l'\''aéronautique."],
    "offer_title": ["Form. Ajusteur Monteur Aéronautique H/F"],
    "offer_description": ["Détails de la formation : La formation se déroulera à Bouguenais pendant 6 mois. \nThéorie en centre de formation 3 mois : La préparation de l'\''assemblage de structures aéronefs : \nVérifier l'\''approvisionnement du matériel, outils, composants nécessaires aux opérations de montage d'\''éléments mécaniques. Ajuster les portées d'\''un élément sur une structure suivant un ou plusieurs plans [...]"]
  }'

[
  {
    "label": "publish",
    "model": "2026-02-20",
    "scores": {
      "publish": 1.0,
      "unpublish": 0.0
    }
  }
]
```
#### Evaluate models

```shell
$ curl http://127.0.0.1:8000/model/evaluate -X POST -H 'Content-Type: application/json' -d '{"versions":["2026-02-20", "2026-02-20"]}'
```

### Exit virtual environment

```shell
$ deactivate
```

## 2. Docker

### Docker Compose (Development)

**Note importante** : La commande `make dev-up` utilise `Dockerfile.local` qui installe `requirements-local.txt` (compatible avec toutes les plateformes : Linux, macOS, Windows - CPU-only). Pour la production avec GPU, utilisez `Dockerfile` avec support CUDA.

```shell
# Start development service with live reload
$ make dev-up

# View logs
$ make logs

# Stop services
$ make down
```

### Build production image (with CUDA support)

```shell
$ docker buildx build --platform linux/amd64 -t lba-classifier .
```

### Run production image

```shell
docker run --rm -it -p 8000:8000 --name classifier lba-classifier
```

### Test docker endpoint

```shell
$ curl http://127.0.0.1:8000/model/score -X POST -H 'Content-Type: application/json' -d '{"text": "Développeur / Développeuse web"}'

{"label":"entreprise","model":"2025-12-18","scores":{"cfa":0.2387,"entreprise":0.4857,"entreprise_cfa":0.2756},"text":"Développeur / Développeuse web"}
```

## API Endpoints

### Health

- **GET /** - Health check endpoint
- **GET /model/version** - Returns current model version

### Model Management

- **GET /model/load?version=YYYY-MM-DD** - Load a specific model version

### Inference

- **POST /model/score** - Classify single text
  - Body: `{"text": "...", "version": "YYYY-MM-DD"}` (version optional)
  - Returns: `{"label": "...", "scores": {...}, "model": "...", "text": "..."}`

- **POST /model/scores** - Batch classify texts (optimized for GPU)
  - Body: `{"items": [{"id": "1", "text": "..."}, ...], "version": "YYYY-MM-DD"}` (version optional)
  - Returns: Array of classification results

### Training

- **POST /model/train** - Train model from online endpoint
  - Body: `{"version": "...", "endpoint": "..."}`

### Evaluation

- **POST /model/evaluate** - Evaluate multiple model versions against the validation dataset
  - Body: `{"versions": [...]}` (minimum 2 versions required)

## Features

- **Automatic model loading** : Le dernier modèle disponible est chargé automatiquement au démarrage
- **Model version control** : Support de versions multiples de modèles
- **Batch processing** : Traitement par batch optimisé pour GPU
- **Error handling** : Contrôle de disponibilité du modèle sur les routes d'inférence
- **Modular architecture** : Code organisé par responsabilité fonctionnelle
