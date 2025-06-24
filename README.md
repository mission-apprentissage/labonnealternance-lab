![](https://avatars1.githubusercontent.com/u/63645182?s=200&v=4)

# Mission Apprentissage Template Repository

- [Mission Apprentissage Template Repository](#mission-apprentissage-template-repository)
  - [Pré-requis](#pré-requis)
  - [Démarrage](#démarrage)
  - [Création du projet](#création-du-projet)
    - [Sentry](#sentry)
    - [MongoDB](#mongodb)
    - [Postgres](#postgres)
    - [Env Ini](#env-ini)
    - [Binary config](#binary-config)
    - [Secrets](#secrets)
    - [UI Config](#ui-config)
    - [Other Files](#other-files)
    - [Mailpit](#mailpit)
    - [Legal](#legal)
    - [Remplacement du README.md](#remplacement-du-readmemd)
    - [Seed](#seed)
    - [Github Actions](#github-actions)
    - [Organisation Secrets](#organisation-secrets)
    - [Mettre à jour les snapshots](#mettre-à-jour-les-snapshots)
    - [Infrastructure](#infrastructure)
    - [Initial Version](#initial-version)
    - [Monitoring](#monitoring)
    - [GitGuardian](#gitguardian)
    - [Github Settings](#github-settings)
    - [Shodan](#shodan)

## Pré-requis

[Suivre la documentation dédiée](./docs/developpement/pre-requesites.md)

## Démarrage

[Créez un nouveau repository en utilisant ce template](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-repository-from-a-template)

Clonez le nouveau repository locallement.

Récupérez manuellement les fichiers de git-lfs depuis le template

```bash
git lfs fetch https://github.com/mission-apprentissage/template-apprentissage.git
git lfs pull
```

Récupérez le vault depuis 1password (assurez vous d'etre connecter via `op account get`)

```bash
.bin/product vault:init
```

Vérifiez que vous pouvez déchiffrer le vault via `gpg --list-packets .infra/vault/.vault-password.gpg`

> Si vous n'etes pas autorisé à déchiffrer le vault, il faudra se rapprocher des personnes habilitées afin d'avoir vos accès ajoutés.

Initialisation de l'envrionnement `yarn setup`

Vous pouvez lancer le projet locallement via `yarn dev`

Le setup va intialiser la base de donnée, il faut maintenant mettre à jour.

Vous pouvez maintenant lancer l'application locallement en [suivant la documentation dédiée](./docs/developpement/developpement.md).

- TODO: GitGuardian, Shodan, Slack Webhook, Uptime

## Création du projet

**Pour une simplification de la procedure, nous allons supposer que vous souhaitez créer le produit nommé `api`**

### Sentry

Il faut créer 1 projet pour l'UI et un pour le Server sur https://sentry.apprentissage.beta.gouv.fr/organizations/sentry/projects/new/

### MongoDB

Créer 2 instances de MongoDB sur OVH pour recette & production.

Pour chaque instance créer 2 utilisateurs:

- `app` avec le role `dbOwner` sur `api`
- `metabase` avec le role `read` sur `api`

Ajouter l'IP de votre server dans les IPs autorisées

### Postgres

Créer 2 instances de Postgres sur OVH pour recette et production.

Ajouter l'IP de votre server dans les IPs autorisées

### Env Ini

Mettre à jour le fichier `.infra/env.ini`

- `product_name`: le nom du produit `api`
- `repo_name`: le nom du repository `api-apprentissage`
- `database_name`: le nom de la BDD `api`
- `domain_name`: le nom du domaine `api`

### Binary config

Mettre à jour le fichier `.bin/product-meta.sh` avec les memes valeurs que le fichier `.infra/env.ini`

### Secrets

Mise à jour du vault `yarn vault:edit`

Mettre à jour les secrets suivants:

- `SERVER_SENTRY_DSN`: le DSN du sentry serveur
- `EMAIL`: addresse email utilisée pour envoyer des emails (créer un alias sur https://admin.alwaysdata.com/mailbox/). Utiliser une addresse de type `nepasrepondre-api@apprentissage.beta.gouv.fr`
- `METABASE_EMAIL_FROM_ADDRESS`: Utiliser la meme que `EMAIL`
- `METABASE_EMAIL_FROM_NAME`: Mettre à jour
- `SEED_GPG_PASSPHRASE`: Générer un nouveau secret `pwgen -s 120 1`
- `MONGODB_KEYFILE`: Générer un nouveau secret `pwgen -s 120 1`

Pour chaque environnement:

- `PUBLIC_URL`: Le nom de domaine public. Pour la preview utiliser le format prédéfini
- `MONGODB_URI`: L'url de connexion de l'utilisateur `app`.
  - Pour `recette` & `production` bien faire attention à la DB à utiliser dans la connection string et d'avoir `authSource=admin`
  - Pour `preview` remplacer simplement la string `TODO_REPLACE_BY_MONGOKEYFILE` par la valeur `MONGODB_KEYFILE`
- `MONGODB_METABASE_URI`: L'url de connexion de l'utilisateur `metabase`. Pour `preview` laisser vide car il n'y a pas de preview
- `AUTH_USER_JWT_SECRET`: Générer un nouveau secret `pwgen -s 120 1`
- `AUTH_PASSWORD_JWT_SECRET`: Générer un nouveau secret `pwgen -s 120 1`
- `SESSION_SECRET`: Générer un nouveau secret `pwgen -s 120 1`
- `SMTP_WEBHOOK_KEY`: Générer un nouveau secret `pwgen -s 64 1`
- `METABASE_ADMIN_EMAIL`: L'addresse email du compte admin. Pour `preview` laisser vide car il n'y a pas de preview
- `METABASE_ADMIN_PASS`: Générer un nouveau secret `pwgen -s -y 64 1`. Pour `preview` laisser vide car il n'y a pas de preview
- `METABASE_DB_URI`: L'url de connexion à la Postgres.. Pour `preview` laisser vide car il n'y a pas de preview
- `METABASE_ENCRYPTION_SECRET_KEY`: Générer un nouveau secret `pwgen -s 120 1`. Pour `preview` laisser vide car il n'y a pas de preview

Pour recette & preview uniquement:

- `SMTP_AUTH_USER`: `user`
- `SMTP_AUTH_PASS`: Générer un nouveau secret `pwgen -s 120 1`, utilisez le **MÊME** password pour les deux environnements.

### UI Config

Mettre à jour le fichier `ui/config.public.ts`

- `sentry.dsn`: le DSN du sentry ui
- `host`: Pour chaque environnement mettre le domaine associé

### Other Files

- `.infra/files/configs/metabase/setup-metabase.sh`
- `server/src/config.ts`: La valeur de `cookieName`
- `server/.env.test`
- `server/tests/globalSetup.ts`
- `server/tests/routes/users.route.test.ts`
- `shared/helpers/openapi/generateOpenapi.test.ts`
- `shared/routes/core.routes.ts`
- `ui/.env.test`

### Mailpit

Les environnements de recette & preview utilisent Mailpit. Il est nécessaire de créer le fichier d'authentification via la commande:

```bash
htpasswd -B .infra/files/configs/mailpit/auth user
```

Entrez le mot passe correpsondant à `SMTP_AUTH_PASS`.

### Legal

Revoir les pages:

- `/`
- `/accessibilite`
- `/cgu`
- `/donnees-personnelles`
- `/mentions-legales`
- `/politique-confidentialite`

### Remplacement du README.md

Remplacer le fichier `README.md` par `PROJECT_README.md`

### Seed

Le seed doit etre regénéré avec votre nouvelle passphrase. Pour cela faire `yarn seed:update`

### Github Actions

1. Décommenter les github actions `.github/workflows`
2. Changer le nom des images docker dans `.github/workflows/release.yml` (`tmpl_xx`)
3. Créer le Github Repository Secret `SLACK_WEBHOOK` (https://api.slack.com/apps/A01JENR8874)

### Organisation Secrets

Autorisez votre repository à accéder aux secrets d'organisation depuis https://github.com/organizations/mission-apprentissage/settings/secrets/actions

- DEPLOY_PASS
- DEPLOY_SSH_PRIVATE_KEY

### Mettre à jour les snapshots

`yarn test`

### Infrastructure

Demander à l'équipe transverse de provisionner les environnements https://github.com/mission-apprentissage/infra/blob/main/docs/provisionning.md

> Une fois les environnements provisionnés et les accès défini veuillez mettre à jour le vault via `yarn vault:edit` puis refermer directement.

Mettez à jour le fichier `.infra/env.ini` avec les IPs des différents serveurs.

### Initial Version

Une fois que les modification ci-dessus ont été réalisé, faites votre première PR et mergez la dans main.

La version `1.0.0` de votre projet sera créé par la Github Action Release.

### Monitoring

Ajoutez votre projet dans le monitoring, pour cela vous référez à https://github.com/mission-apprentissage/monitoring/blob/main/README.md

### GitGuardian

Ajoutez votre projet sur Gitguardian https://dashboard.gitguardian.com/workspace/220324/settings/workspace/integrations/github

### Github Settings

- General:
  - Features: `Issues` only
  - Always suggest updating pull request branches: `on`
  - Allow auto-merge: `on`
  - Automatically delete head branches : `on`
- Branches:
  - `main` branch protection
    - Require a pull request before merging:
      - Require approvals: `on`
    - Require status checks to pass before merging
      - `tests / Tests`
    - Require merge queue:
      - Merge method: `Squash & merge`
- Code security and analysis
  - Private vulnerability reporting: `on`
  - Dependency graph: `on`
  - Dependabot:
    - Dependabot alerts: `on`
    - Dependabot security updates: `on`
    - Grouped security updates: `on`
  - Code scanning:
    - CodeQL analysis: Setup Default
  - Secret scanning: `on`
    - Push protection: `on`

### Shodan

Créez un compte [Shodan](https://shodan.io) en tant que `Member`.

Ajoutez la config Slack Webhook pour recevoir les alertes.

Ajoutez vos urls à monitorer avec notification Slack:

- <product_name>.apprentissage.beta.gouv.fr
- <product_name>-recette.apprentissage.beta.gouv.fr
- <product_name>-preview.apprentissage.beta.gouv.fr
