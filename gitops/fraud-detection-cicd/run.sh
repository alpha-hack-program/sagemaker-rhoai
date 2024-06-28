#!/bin/sh
ARGOCD_APP_NAME=fraud-cicd

# Load environment variables
. ../../bootstrap/.env

helm template . --name-template ${ARGOCD_APP_NAME} \
  --set instanceName="${INSTANCE_NAME}" \
  --set dataScienceProjectNamespace=${DATA_SCIENCE_PROJECT_NAMESPACE} \
  --set dataScienceProjectDisplayName=${DATA_SCIENCE_PROJECT_NAMESPACE} \
  --include-crds