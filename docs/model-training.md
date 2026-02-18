# Documentation : Modèle d'entrainement

## Vue d'ensemble

Le service classifie les textes d'offres d'emploi en 3 catégories :

- **`cfa`** — offre publiée par un centre de formation
- **`entreprise`** — offre publiée par une entreprise
- **`entreprise_cfa`** — offre publiée par une entreprise également centre de formation

---

## Pipeline d'entrainement

Le pipeline se décompose en 4 étapes successives :

```
Données brutes (API)
       ↓
  Undersampling       ← équilibrage des classes
       ↓
  Embeddings          ← CamemBERT transforme le texte en vecteurs numériques
       ↓
  PCA + StandardScaler ← réduction de dimension et normalisation
       ↓
  SVM (SVC)           ← classification finale
```

### 1. Collecte des données

Les données sont récupérées depuis l'endpoint LBA (`/api/classification`). Chaque offre contient :

| Champ | Rôle |
|---|---|
| `workplace_name` | Nom de l'entreprise |
| `workplace_description` | Description de l'entreprise |
| `offer_title` | Titre de l'offre |
| `offer_description` | Description du poste |

Ces 4 champs sont **concaténés** pour former le texte d'entrée du modèle :
```
workplace_name + "\n" + workplace_description + "\n" + offer_title + "\n" + offer_description
```

### 2. Undersampling (équilibrage des classes)

Avant l'encodage, les classes sont rééquilibrées en sous-échantillonnant les classes sur-représentées pour matcher la taille de la classe la moins représentée.

**Exemple :**
```
Avant : cfa=252, entreprise=659, entreprise_cfa=509
Après : cfa=252, entreprise=252, entreprise_cfa=252  (total: 756)
```

> Cette étape est critique. Sans elle, le modèle apprend à prédire majoritairement la classe dominante et ignore les autres.

### 3. Encodage en embeddings (CamemBERT)

Chaque texte est transformé en un vecteur de 768 dimensions par le modèle de langage `almanach/camembertav2-base` (CamemBERT v2, spécialisé français).

L'embedding est calculé comme la **moyenne des représentations de tous les tokens**, puis normalisé à longueur unitaire.

### 4. PCA + Normalisation

Avant la classification, les embeddings passent par un pipeline scikit-learn :

1. **`SimpleImputer`** — remplace les valeurs manquantes par la médiane
2. **`StandardScaler`** — centre et normalise les features
3. **`PCA`** — réduit la dimension en conservant **99,99% de la variance** expliquée

Le nombre de composantes PCA est déterminé automatiquement à chaque entrainement.

### 5. Classifieur SVM

Le modèle final est un **Support Vector Machine** avec noyau RBF (`rbf`), qui sépare les 3 classes dans l'espace réduit par PCA.

Paramètres fixes : `kernel='rbf'`, `probability=True`, `random_state=42`.

---

## Ce qui fait varier la qualité du modèle

### Volume de données

Plus il y a d'offres étiquetées par classe, meilleure est la généralisation. Le modèle actuel est limité par le nombre d'offres `cfa` disponibles (classe minoritaire).

| Situation | Effet |
|---|---|
| Peu de données (< 200/classe) | Modèle peu généralisable, scores instables |
| Données équilibrées (≥ 500/classe) | Meilleure discrimination entre les classes |

### Équilibre des classes

Le déséquilibre est le facteur le plus impactant. Si une classe représente 50%+ des données, le modèle tend à la prédire par défaut (biais de majorité).

**Solution appliquée :** undersampling dynamique — automatiquement appliqué à chaque entrainement.

### Qualité des textes

Le modèle est sensible au contenu textuel des offres :

- Des textes vides ou très courts dégradent la précision
- Des textes avec du HTML non nettoyé introduisent du bruit
- Les 4 champs contribuent au signal — leur absence réduit la qualité

### Séparation entrainement / test

Le split est 80% entrainement / 20% test, stratifié par classe (`stratify=label_df`). Le `random_state=42` garantit la reproductibilité.

---

## Scores : comment les interpréter

Lors d'un entrainement, deux scores sont retournés :

```json
{"train_score": 0.83, "test_score": 0.78}
```

| Score | Description |
|---|---|
| `train_score` | Précision sur les données d'entrainement (80%) |
| `test_score` | Précision sur les données de test (20%) |

Ces scores sont calculés sur les **mêmes données que l'entrainement** (même endpoint, même distribution). Ils mesurent la capacité du modèle à apprendre, pas sa généralisation réelle.

> Un `test_score` élevé (> 0.80) ne garantit pas de bonnes performances sur des cas réels si les données d'entrainement ne sont pas représentatives.

### Dataset de validation humain

Pour mesurer la performance réelle, le endpoint `/model/evaluate` compare plusieurs versions sur un jeu de données **validé manuellement** (`server/data/validation_dataset.json`).

Ce dataset contient 25 offres annotées à la main. Les métriques retournées sont :

- **`accuracy`** — taux de prédiction correcte
- **`f1`** — F1-score pondéré (plus robuste sur données déséquilibrées)

---

## Bonnes pratiques

### Toujours comparer sur le dataset humain

Le dataset de validation manuel (`/model/evaluate`) est la seule mesure fiable. Comparer systématiquement une nouvelle version avec la version en production avant de déployer.

```shell
curl http://127.0.0.1:8000/model/evaluate -X POST \
  -H 'Content-Type: application/json' \
  -d '{"versions": ["version-prod", "version-nouvelle"]}'
```

### Alimenter le dataset de validation au fil du temps

Ajouter régulièrement de nouvelles offres vérifiées par des humains dans `server/data/validation_dataset.json`. Plus ce dataset est représentatif et large, plus la comparaison de modèles est fiable.

### Réentrainer quand les données évoluent

Le modèle doit être réentrainé si :
- Le nombre d'offres disponibles dans l'endpoint augmente significativement
- La distribution des labels change dans les données sources
- La qualité des prédictions se dégrade sur de nouvelles offres

### Nommer les versions par date

Les versions suivent le format `YYYY-MM-DD` (ex: `2025-12-18`). Cela permet de trier chronologiquement et de retrouver la version de production facilement.

---

## Paramètres fixes

Ces paramètres sont fixes dans le code et ne varient pas entre les entrainements :

| Paramètre | Valeur | Raison |
|---|---|---|
| `random_state` | `42` | Reproductibilité du split et du SVM |
| `test_size` | `0.2` | Split 80/20 standard |
| `pca_threshold` | `0.9999` | Conservation de 99,99% de la variance |
| `lang_model` | `almanach/camembertav2-base` | Meilleur modèle français disponible |
| `svc_kernel` | `rbf` | Adapté aux espaces de haute dimension |
| `batch_size` | `20` | Encodage par batch pour la mémoire GPU |

---

## Résumé visuel

```
API endpoint
    │
    ▼
Offres brutes (workplace_name + workplace_description + offer_title + offer_description)
    │
    ▼
Undersampling → équilibrer cfa / entreprise / entreprise_cfa au nombre minimal
    │
    ▼
CamemBERT (almanach/camembertav2-base) → vecteur 768 dimensions par offre
    │
    ▼
Pipeline sklearn: SimpleImputer → StandardScaler → PCA (99.99% variance)
    │
    ▼
SVM (kernel RBF) → prédiction : "cfa" | "entreprise" | "entreprise_cfa"
    │
    ▼
Sauvegarde sur HuggingFace Hub (modèle + dataset)
```
