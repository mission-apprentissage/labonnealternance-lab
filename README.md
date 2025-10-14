![](https://avatars1.githubusercontent.com/u/63645182?s=200&v=4)

# La bonne alternance - Laboratoire

## Fiche Produit

## Documentation

## 1. Test application
### Install requirements
```shell
$ cd server && python -m venv .venv && source .venv/bin/activate
$ pip install -r requirements.txt
```

### Running development server
```shell
$ python main.py
```

### Test endpoint
#### Check API status
```shell
$ curl http://127.0.0.1:8000/

{"status":"LBA classifier API ready."}
```

#### Load model version
```shell
curl http://127.0.0.1:8000/model/load?version='2025-08-06'

{"model":"2025-08-06"}
```

#### Check model version
```shell
$ curl http://127.0.0.1:8000/model/version

{"model":"2025-08-06"}
```

#### Train model version
```shell
$ curl http://127.0.0.1:8000/model/train -X POST -H 'Content-Type: application/json' -d '{"version": "2025-09-18", "ids":["1","2","3","4","5","6","7","8","9","10","11","12"], "texts": ["texte 1","texte 2","texte 3","texte 4","texte 5","texte 6","texte 7","texte 8","texte 9","texte 10","texte 11","texte 12"], "labels": ["cfa","cfa_entreprise","entreprise","cfa_entreprise","entreprise","cfa","cfa_entreprise","entreprise","cfa","entreprise","cfa_entreprise","entreprise"]}'

{"dataset_url":"https://huggingface.co/datasets/la-bonne-alternance/2025-09-18","model_url":"https://huggingface.co/la-bonne-alternance/2025-09-18","test_score":0.3333,"train_score":0.8888,"version":"2025-09-18"}
```

#### Score one text
```shell
$ curl http://127.0.0.1:8000/model/score -X POST -H 'Content-Type: application/json' -d '{"version":"2025-08-06", "text": "DIRECTION INTERMINISTERIELLE DU NUMERIQUE (DINUM)\nService du Premier ministre, placé sous l’autorité du ministre de la Transformation et de la Fonction publiques, la direction interministérielle du numérique (DINUM) a pour mission d’élaborer la stratégie numérique de l’État et de piloter sa mise en œuvre. Notre objectif : un État plus efficace, plus simple et plus souverain grâce au numérique.\nDéveloppeur / Développeuse web\n\nConçoit, développe et met au point un projet d’application informatique, de la phase d’étude à son intégration, pour un client ou une entreprise selon des besoins fonctionnels et un cahier des charges. Peut conduire des projets de développement. Peut coordonner une équipe."}'

{"label":"entreprise","model":"2025-07-28","scores":{"cfa":0.1004,"entreprise":0.5367,"entreprise_cfa":0.3629},"text":"DIRECTION INTERMINISTERIELLE DU NUMERIQUE (DINUM)\nService du Premier ministre, plac\u00e9 sous l\u2019autorit\u00e9 du ministre de la Transformation et de la Fonction publiques, la direction interminist\u00e9rielle du num\u00e9rique (DINUM) a pour mission d\u2019\u00e9laborer la strat\u00e9gie num\u00e9rique de l\u2019\u00c9tat et de piloter sa mise en \u0153uvre. Notre objectif : un \u00c9tat plus efficace, plus simple et plus souverain gr\u00e2ce au num\u00e9rique.\nD\u00e9veloppeur / D\u00e9veloppeuse web\n\nCon\u00e7oit, d\u00e9veloppe et met au point un projet d\u2019application informatique, de la phase d\u2019\u00e9tude \u00e0 son int\u00e9gration, pour un client ou une entreprise selon des besoins fonctionnels et un cahier des charges. Peut conduire des projets de d\u00e9veloppement. Peut coordonner une \u00e9quipe."}

$ "curl http://127.0.0.1:8000/model/score -X POST -H 'Content-Type: application/json' -d '{"version":"2025-09-18", "text": "DIRECTION INTERMINISTERIELLE DU NUMERIQUE (DINUM)\nService du Premier ministre, placé sous l’autorité du ministre de la Transformation et de la Fonction publiques, la direction interministérielle du numérique (DINUM) a pour mission d’élaborer la stratégie numérique de l’État et de piloter sa mise en œuvre. Notre objectif : un État plus efficace, plus simple et plus souverain grâce au numérique.\nDéveloppeur / Développeuse web\n\nConçoit, développe et met au point un projet d’application informatique, de la phase d’étude à son intégration, pour un client ou une entreprise selon des besoins fonctionnels et un cahier des charges. Peut conduire des projets de développement. Peut coordonner une équipe."}'"
```

#### Score multiple texts
```shell
$ curl http://127.0.0.1:8000/model/scores -X POST -H 'Content-Type: application/json' -d '{"version":"2025-08-06", "items": [{"id":"1", "text": "DIRECTION INTERMINISTERIELLE DU NUMERIQUE (DINUM)\nService du Premier ministre, placé sous l’autorité du ministre de la Transformation et de la Fonction publiques, la direction interministérielle du numérique (DINUM) a pour mission d’élaborer la stratégie numérique de l’État et de piloter sa mise en œuvre. Notre objectif : un État plus efficace, plus simple et plus souverain grâce au numérique.\nDéveloppeur / Développeuse web\n\nConçoit, développe et met au point un projet d’application informatique, de la phase d’étude à son intégration, pour un client ou une entreprise selon des besoins fonctionnels et un cahier des charges. Peut conduire des projets de développement. Peut coordonner une équipe."}, {"id":"2", "text": "Hively Hospitality - Thalazur\nHIVELY HOSPITALITY est une ruche vivante ! Qui s’adresse à toutes celles et ceux qui sont en quête de réalisation de soi.\nUne ruche où chacun peut apporter sa contribution à un ensemble plus grand et fructueux, un collectif solidaire, en mouvement, qui œuvre à créer ensemble une hospitalité joyeuse, douce et accueillante, pour ses clients comme pour ses collaborateurs.\nAvec 15 marques, 80 établissements en France et en Europe et 60 métiers dans l’hôtellerie, la restauration et le bien-être, HIVELY HOSPITALITY offre une diversité d’opportunités et d’évolutions. Un seul prérequis de départ : l’envie !\nAssistant(e) Gouvernant(e) H/F (alternance)\nNotre hôtel : Thalazur Antibes Votre mission : Prendre soin de nos clients en chouchoutant leurs espaces de vie ! Au quotidien, il s’agira de réaliser :Apprendre la prise en charge de  la gestion du service d’étages en coordonnant, contrôlant et planifiant l’activité de vos équipes. Gérer les matériels, stocks et fournitures des produits d’accueil et d’entretien. Assurer une hygiène parfaite dans les espaces réservés au séjour de la clientèle. Animer et encadrer les équipes des étages. Nous vous proposons :Des tarifs préférentiels dans notre réseau d’établissements Mutuelle et Prévoyance offrant des garanties et une couverture maximales Adhésion à une plateforme web avec réductions sur des produits du quotidien (cinéma, parcs d’attraction, courses alimentaires, salles de sport… Parking gratuit Votre profil :Qualités relationnelles Envie de rejoindre une équipe unie, énergique et professionnelle Nous faisons partie de HIVELY HOSPITALITY. Avec une quinzaine de marques à forte notoriété, 80 établissements en France et en Europe, et 60 métiers dans l’hôtellerie, la restauration et le bien-être, HIVELY HOSPITALITY offre une diversité d’opportunités et d’évolutions à toutes celles et ceux qui sont en quête de réalisation de soi. Un seul prérequis de départ : l’envie !"}]}'

[{"id":"1","label":"entreprise","model":"2025-07-28 offres_ft_svc.pkl","scores":{"cfa":0.2387,"entreprise":0.4857,"entreprise_cfa":0.2756},"text":"DIRECTION INTERMINISTERIELLE DU NUMERIQUE (DINUM)\nService du Premier ministre, plac\u00e9 sous l\u2019autorit\u00e9 du ministre de la Transformation et de la Fonction publiques, la direction interminist\u00e9rielle du num\u00e9rique (DINUM) a pour mission d\u2019\u00e9laborer la strat\u00e9gie num\u00e9rique de l\u2019\u00c9tat et de piloter sa mise en \u0153uvre. Notre objectif : un \u00c9tat plus efficace, plus simple et plus souverain gr\u00e2ce au num\u00e9rique.\nD\u00e9veloppeur / D\u00e9veloppeuse web\n\nCon\u00e7oit, d\u00e9veloppe et met au point un projet d\u2019application informatique, de la phase d\u2019\u00e9tude \u00e0 son int\u00e9gration, pour un client ou une entreprise selon des besoins fonctionnels et un cahier des charges. Peut conduire des projets de d\u00e9veloppement. Peut coordonner une \u00e9quipe."},{"id":"2","label":"entreprise","model":"2025-07-28 offres_ft_svc.pkl","scores":{"cfa":0.2102,"entreprise":0.5886,"entreprise_cfa":0.2012},"text":"Hively Hospitality - Thalazur\nHIVELY HOSPITALITY est une ruche vivante ! Qui s\u2019adresse \u00e0 toutes celles et ceux qui sont en qu\u00eate de r\u00e9alisation de soi.\nUne ruche o\u00f9 chacun peut apporter sa contribution \u00e0 un ensemble plus grand et fructueux, un collectif solidaire, en mouvement, qui \u0153uvre \u00e0 cr\u00e9er ensemble une hospitalit\u00e9 joyeuse, douce et accueillante, pour ses clients comme pour ses collaborateurs.\nAvec 15 marques, 80 \u00e9tablissements en France et en Europe et 60 m\u00e9tiers dans l\u2019h\u00f4tellerie, la restauration et le bien-\u00eatre, HIVELY HOSPITALITY offre une diversit\u00e9 d\u2019opportunit\u00e9s et d\u2019\u00e9volutions. Un seul pr\u00e9requis de d\u00e9part : l\u2019envie !\nAssistant(e) Gouvernant(e) H/F (alternance)\nNotre h\u00f4tel : Thalazur Antibes Votre mission : Prendre soin de nos clients en chouchoutant leurs espaces de vie ! Au quotidien, il s\u2019agira de r\u00e9aliser :Apprendre la prise en charge de  la gestion du service d\u2019\u00e9tages en coordonnant, contr\u00f4lant et planifiant l\u2019activit\u00e9 de vos \u00e9quipes. G\u00e9rer les mat\u00e9riels, stocks et fournitures des produits d\u2019accueil et d\u2019entretien. Assurer une hygi\u00e8ne parfaite dans les espaces r\u00e9serv\u00e9s au s\u00e9jour de la client\u00e8le. Animer et encadrer les \u00e9quipes des \u00e9tages. Nous vous proposons :Des tarifs pr\u00e9f\u00e9rentiels dans notre r\u00e9seau d\u2019\u00e9tablissements Mutuelle et Pr\u00e9voyance offrant des garanties et une couverture maximales Adh\u00e9sion \u00e0 une plateforme web avec r\u00e9ductions sur des produits du quotidien (cin\u00e9ma, parcs d\u2019attraction, courses alimentaires, salles de sport\u2026 Parking gratuit Votre profil :Qualit\u00e9s relationnelles Envie de rejoindre une \u00e9quipe unie, \u00e9nergique et professionnelle Nous faisons partie de HIVELY HOSPITALITY. Avec une quinzaine de marques \u00e0 forte notori\u00e9t\u00e9, 80 \u00e9tablissements en France et en Europe, et 60 m\u00e9tiers dans l\u2019h\u00f4tellerie, la restauration et le bien-\u00eatre, HIVELY HOSPITALITY offre une diversit\u00e9 d\u2019opportunit\u00e9s et d\u2019\u00e9volutions \u00e0 toutes celles et ceux qui sont en qu\u00eate de r\u00e9alisation de soi. Un seul pr\u00e9requis de d\u00e9part : l\u2019envie !"}]
```

#### Evaluate models
```shell
$ curl http://127.0.0.1:8000/model/evaluate -X POST -H 'Content-Type: application/json' -d '{"versions":["2025-08-06", "2025-09-18"], "texts": ["texte 1","texte 2","texte 3","texte 4","texte 5","texte 6","texte 7","texte 8","texte 9","texte 10","texte 11","texte 12"], "labels": ["cfa","cfa_entreprise","entreprise","cfa_entreprise","entreprise","cfa","cfa_entreprise","entreprise","cfa","entreprise","cfa_entreprise","entreprise"]}'
```

### Exit virtual environment
```shell
$ deactivate
```

## 2. Create docker image
### Build image
```shell
$ docker buildx build --platform linux/amd64 -t lba-classifier .
```

### Run image
```shell
docker run --rm -it -p 8000:8000 --name classifier lba-classifier
```

### Test docker endpoint
```shell
$ curl http://172.17.0.2:8000/score -X POST -H 'Content-Type: application/json' -d '{"text": "DIRECTION INTERMINISTERIELLE DU NUMERIQUE (DINUM)\nService du Premier ministre, placé sous l’autorité du ministre de la Transformation et de la Fonction publiques, la direction interministérielle du numérique (DINUM) a pour mission d’élaborer la stratégie numérique de l’État et de piloter sa mise en œuvre. Notre objectif : un État plus efficace, plus simple et plus souverain grâce au numérique.\nDéveloppeur / Développeuse web\n\nConçoit, développe et met au point un projet d’application informatique, de la phase d’étude à son intégration, pour un client ou une entreprise selon des besoins fonctionnels et un cahier des charges. Peut conduire des projets de développement. Peut coordonner une équipe."}'

{"label":"entreprise","model":"2025-07-28 offres_ft_svc.pkl","scores":{"cfa":0.2387,"entreprise":0.4857,"entreprise_cfa":0.2756},"text":"DIRECTION INTERMINISTERIELLE DU NUMERIQUE (DINUM)\nService du Premier ministre, plac\u00e9 sous l\u2019autorit\u00e9 du ministre de la Transformation et de la Fonction publiques, la direction interminist\u00e9rielle du num\u00e9rique (DINUM) a pour mission d\u2019\u00e9laborer la strat\u00e9gie num\u00e9rique de l\u2019\u00c9tat et de piloter sa mise en \u0153uvre. Notre objectif : un \u00c9tat plus efficace, plus simple et plus souverain gr\u00e2ce au num\u00e9rique.\nD\u00e9veloppeur / D\u00e9veloppeuse web\n\nCon\u00e7oit, d\u00e9veloppe et met au point un projet d\u2019application informatique, de la phase d\u2019\u00e9tude \u00e0 son int\u00e9gration, pour un client ou une entreprise selon des besoins fonctionnels et un cahier des charges. Peut conduire des projets de d\u00e9veloppement. Peut coordonner une \u00e9quipe."}
```
