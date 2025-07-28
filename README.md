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
```shell
$ curl http://127.0.0.1:5000/score -X POST -H 'Content-Type: application/json' -d '{"text": "DIRECTION INTERMINISTERIELLE DU NUMERIQUE (DINUM)\nService du Premier ministre, placé sous l’autorité du ministre de la Transformation et de la Fonction publiques, la direction interministérielle du numérique (DINUM) a pour mission d’élaborer la stratégie numérique de l’État et de piloter sa mise en œuvre. Notre objectif : un État plus efficace, plus simple et plus souverain grâce au numérique.\nDéveloppeur / Développeuse web\n\nConçoit, développe et met au point un projet d’application informatique, de la phase d’étude à son intégration, pour un client ou une entreprise selon des besoins fonctionnels et un cahier des charges. Peut conduire des projets de développement. Peut coordonner une équipe."}'

{"label":"entreprise","model":"2025-07-28 offres_ft_svc.pkl","scores":{"cfa":0.2387,"entreprise":0.4857,"entreprise_cfa":0.2756},"text":"DIRECTION INTERMINISTERIELLE DU NUMERIQUE (DINUM)\nService du Premier ministre, plac\u00e9 sous l\u2019autorit\u00e9 du ministre de la Transformation et de la Fonction publiques, la direction interminist\u00e9rielle du num\u00e9rique (DINUM) a pour mission d\u2019\u00e9laborer la strat\u00e9gie num\u00e9rique de l\u2019\u00c9tat et de piloter sa mise en \u0153uvre. Notre objectif : un \u00c9tat plus efficace, plus simple et plus souverain gr\u00e2ce au num\u00e9rique.\nD\u00e9veloppeur / D\u00e9veloppeuse web\n\nCon\u00e7oit, d\u00e9veloppe et met au point un projet d\u2019application informatique, de la phase d\u2019\u00e9tude \u00e0 son int\u00e9gration, pour un client ou une entreprise selon des besoins fonctionnels et un cahier des charges. Peut conduire des projets de d\u00e9veloppement. Peut coordonner une \u00e9quipe."}
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
$ curl http://172.17.0.2:5000/score -X POST -H 'Content-Type: application/json' -d '{"text": "DIRECTION INTERMINISTERIELLE DU NUMERIQUE (DINUM)\nService du Premier ministre, placé sous l’autorité du ministre de la Transformation et de la Fonction publiques, la direction interministérielle du numérique (DINUM) a pour mission d’élaborer la stratégie numérique de l’État et de piloter sa mise en œuvre. Notre objectif : un État plus efficace, plus simple et plus souverain grâce au numérique.\nDéveloppeur / Développeuse web\n\nConçoit, développe et met au point un projet d’application informatique, de la phase d’étude à son intégration, pour un client ou une entreprise selon des besoins fonctionnels et un cahier des charges. Peut conduire des projets de développement. Peut coordonner une équipe."}'

{"label":"entreprise","model":"2025-07-28 offres_ft_svc.pkl","scores":{"cfa":0.2387,"entreprise":0.4857,"entreprise_cfa":0.2756},"text":"DIRECTION INTERMINISTERIELLE DU NUMERIQUE (DINUM)\nService du Premier ministre, plac\u00e9 sous l\u2019autorit\u00e9 du ministre de la Transformation et de la Fonction publiques, la direction interminist\u00e9rielle du num\u00e9rique (DINUM) a pour mission d\u2019\u00e9laborer la strat\u00e9gie num\u00e9rique de l\u2019\u00c9tat et de piloter sa mise en \u0153uvre. Notre objectif : un \u00c9tat plus efficace, plus simple et plus souverain gr\u00e2ce au num\u00e9rique.\nD\u00e9veloppeur / D\u00e9veloppeuse web\n\nCon\u00e7oit, d\u00e9veloppe et met au point un projet d\u2019application informatique, de la phase d\u2019\u00e9tude \u00e0 son int\u00e9gration, pour un client ou une entreprise selon des besoins fonctionnels et un cahier des charges. Peut conduire des projets de d\u00e9veloppement. Peut coordonner une \u00e9quipe."}
```
