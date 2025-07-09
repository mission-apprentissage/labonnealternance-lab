![](https://avatars1.githubusercontent.com/u/63645182?s=200&v=4)

# La bonne alternance - Laboratoire

## Fiche Produit

## Documentation


### Create virtual environment
```shell
$ python -m venv .venv
$ source .venv/bin/activate
```
### Running development server
```shell
$ cd server
$ pip install -r requirements.txt
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



