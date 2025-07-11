#!/usr/bin/env bash
set -euo pipefail

echo "Updating local server/.env"
ANSIBLE_CONFIG="${ROOT_DIR}/.infra/ansible/ansible.cfg" ansible all \
  --limit "local" \
  -m template \
  -a "src=\"${ROOT_DIR}/.infra/.env_server\" dest=\"${ROOT_DIR}/server/.env\"" \
  --extra-vars "@${ROOT_DIR}/.infra/vault/vault.yml" \
  --vault-password-file="${SCRIPT_DIR}/get-vault-password-client.sh"

echo "PUBLIC_VERSION=0.0.0-local" >> "${ROOT_DIR}/server/.env"
echo "COMMIT_HASH=$(git rev-parse --short HEAD)" >> "${ROOT_DIR}/server/.env"
