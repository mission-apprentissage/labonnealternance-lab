![](https://avatars1.githubusercontent.com/u/63645182?s=200&v=4)

# La bonne alternance - Laboratoire

## Fiche Produit

## Documentation

## 1. Add environment variable
The application depends on this secret environment variables:
- $LAB_HF_TOKEN

## 2. Test application
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
$ curl http://127.0.0.1:5000/score -X POST -H 'Content-Type: application/json' -d '{"text": "CFA boulangerie"}'
{"label":"cfa","scores":{"cfa":0.36,"entreprise":0.32,"entreprise_cfa":0.32},"text":"CFA boulangerie"}
```

### Exit virtual environment
```shell
$ deactivate
```

## 3. Create docker image
### Build image
```shell
$ docker buildx build --platform linux/amd64 -t lba-classifier .
```

### Run image
```shell
docker run --rm -it -p 8000:8000 -e HF_TOKEN="$LAB_HF_TOKEN" --name classifier lba-classifier
```

### Test docker endpoint
```shell
$ curl http://172.17.0.2:5000/score -X POST -H 'Content-Type: application/json' -d '{"text": "CFA boulangerie"}'
{"label":"cfa","scores":{"cfa":0.36,"entreprise":0.32,"entreprise_cfa":0.32},"text":"CFA boulangerie"}
```
