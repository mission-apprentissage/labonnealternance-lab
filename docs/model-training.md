# Documentation : Modèle d'entrainement

## Vue d'ensemble

Le service classifie les offres d'emploi en 2 catégories :

- **`publish`** — offre publiable (entreprise ou entreprise_cfa)
- **`unpublish`** — offre non publiable (cfa)

---

## Pipeline d'entrainement

```
Données brutes (API)
       ↓
  Mapping des labels      ← cfa → unpublish, reste → publish
       ↓
  Split offer_description ← découpe la description en 2 moitiés
       ↓
  Embeddings (x5 champs)  ← SentenceTransformer encode chaque champ séparément
       ↓
  PCA + StandardScaler    ← réduction de dimension et normalisation
       ↓
  SMOTE                   ← surééchantillonnage de la classe minoritaire
       ↓
  Logistic Regression     ← classification finale
```

### 1. Collecte des données

Les données sont récupérées depuis l'endpoint LBA (`/api/classification`). Chaque offre contient :

| Champ | Rôle |
|---|---|
| `workplace_name` | Nom de l'entreprise |
| `workplace_description` | Description de l'entreprise |
| `offer_title` | Titre de l'offre |
| `offer_description` | Description du poste |

### 2. Transformation des labels

Les labels sont remappés en binaire :

```
cfa           → unpublish
entreprise    → publish
entreprise_cfa → publish
```

### 3. Découpe de la description

`offer_description` est découpée en deux moitiés (`offer_description_1`, `offer_description_2`). Le modèle encode donc **5 champs** au total, capturant séparément le début et la fin de la description.

### 4. Encodage en embeddings (SentenceTransformer)

Chaque champ est encodé **séparément** par le modèle de langage (`almanach/camembertav2-base` via SentenceTransformer), produisant un vecteur de 768 dimensions par champ.

Au total : **5 champs × 768 dimensions = 3840 features** par offre.

> L'encodage séparé de chaque champ permet au modèle de distinguer l'importance relative du nom de l'entreprise, de la description, et du titre/description de l'offre.

### 5. PCA + Normalisation

Avant la classification, les features passent par un pipeline sklearn :

1. **`SimpleImputer`** — remplace les valeurs manquantes par la médiane
2. **`StandardScaler`** — centre et normalise les features
3. **`PCA`** — réduit la dimension en conservant **99,99% de la variance** expliquée

Le nombre de composantes PCA est déterminé automatiquement à chaque entrainement.

### 6. SMOTE (équilibrage des classes)

SMOTE (**S**ynthetic **M**inority **O**versampling **TE**chnique) génère des exemples synthétiques pour la classe minoritaire afin d'équilibrer les classes.

Contrairement à l'undersampling qui supprime des exemples, SMOTE **crée de nouveaux exemples interpolés** dans l'espace des features, préservant ainsi l'intégralité des données originales.

> SMOTE est appliqué uniquement sur les données d'entrainement (dans le pipeline), jamais sur les données de test.

### 7. Classifieur — Régression Logistique

Le classifieur final est une **Régression Logistique** (`LogisticRegression`), qui produit des probabilités pour chaque classe (`publish`, `unpublish`).

Paramètres : `random_state=42`, `max_iter=1000`.

---

## Ce qui fait varier la qualité du modèle

### Volume de données

Plus il y a d'offres étiquetées, meilleure est la généralisation. La classe `unpublish` (cfa) est généralement la classe minoritaire — SMOTE compensera, mais plus de données réelles restent préférables.

### Équilibre des classes

Le déséquilibre est géré automatiquement par SMOTE. Cependant, un déséquilibre trop fort (ex. 10x) peut dégrader la qualité des exemples synthétiques générés.

### Qualité des textes

- Des champs vides (`workplace_description`, `offer_description`) réduisent l'information disponible
- Les 5 champs encodés contribuent chacun au signal — leur absence partielle est tolérée mais dégrade la précision
- Les textes avec du HTML ou des espaces parasites introduisent du bruit

### Séparation entrainement / test

Le split est **80% entrainement / 20% test**, stratifié par classe (`stratify=labels`). Le `random_state=42` garantit la reproductibilité.

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

Ces scores mesurent la capacité du modèle à apprendre sur les données de l'endpoint. Ils ne reflètent pas nécessairement les performances sur des cas réels.

> Un écart important entre `train_score` et `test_score` (> 0.10) indique du surapprentissage.

### Dataset de validation humain

Le endpoint `/model/evaluate` mesure les performances sur un jeu de données **validé manuellement** (`server/data/validation-dataset.json`).

Les métriques retournées :

- **`accuracy`** — taux de prédiction correcte
- **`f1`** — F1-score pondéré (plus robuste sur données déséquilibrées)

---

## Bonnes pratiques

### Toujours valider sur le dataset humain

Le dataset de validation manuel est la seule mesure fiable de la qualité réelle. Comparer systématiquement une nouvelle version avant de déployer.

```shell
curl http://127.0.0.1:8000/model/evaluate -X POST \
  -H 'Content-Type: application/json' \
  -d '{"versions": ["version-prod", "version-nouvelle"]}'
```

### Alimenter le dataset de validation au fil du temps

Ajouter régulièrement de nouvelles offres vérifiées par des humains dans `server/data/validation-dataset.json`. Plus ce dataset est représentatif, plus la comparaison de modèles est fiable.

### Réentrainer quand les données évoluent

Le modèle doit être réentrainé si :
- Le volume de données de l'endpoint augmente significativement
- La distribution des labels change dans les données sources
- La qualité des prédictions se dégrade sur de nouvelles offres

### Nommer les versions par date

Les versions suivent le format `YYYY-MM-DD` (ex: `2026-02-20`). Cela permet de trier chronologiquement et de retrouver la version en production facilement.

---

## Paramètres fixes

| Paramètre | Valeur | Raison |
|---|---|---|
| `random_state` | `42` | Reproductibilité du split, SMOTE, et LR |
| `test_size` | `0.2` | Split 80/20 standard |
| `pca_threshold` | `0.9999` | Conservation de 99,99% de la variance |
| `lang_model` | `almanach/camembertav2-base` | Meilleur modèle français disponible |
| `batch_size` | `32` | Encodage par batch pour la mémoire GPU |
| `max_iter` | `1000` | Assure la convergence de la régression logistique |
| `features` | `3840` | 5 champs × 768 dimensions par embedding |

---

## Résumé visuel

```
API endpoint  →  offres avec labels : cfa / entreprise / entreprise_cfa
                                               ↓
                              Remapping : cfa → unpublish
                                          reste → publish
                                               ↓
                     Split offer_description en 2 moitiés
                                               ↓
         SentenceTransformer (almanach/camembertav2-base)
         encode 5 champs séparément → 5 × 768 = 3840 features
                                               ↓
         Pipeline sklearn:
           SimpleImputer → StandardScaler → PCA (99.99% variance)
                                               ↓
                        SMOTE (oversampling classe minoritaire)
                                               ↓
                Logistic Regression → "publish" | "unpublish"
                                               ↓
                  Sauvegarde sur HuggingFace Hub
```
