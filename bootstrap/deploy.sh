#!/bin/sh

# Load environment variables
. .env

# Create an ArgoCD application to deploy the helm chart at this repository and path ./gitops/doc-bot
cat <<EOF | kubectl apply -f -
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: fraud-detection
  namespace: openshift-gitops
  annotations:
    argocd.argoproj.io/compare-options: IgnoreExtraneous
spec:
  project: default
  destination:
    server: 'https://kubernetes.default.svc'
    namespace: ${DATA_SCIENCE_PROJECT_NAMESPACE}
  source:
    path: gitops/fruad-detection
    repoURL: ${REPO_URL}
    targetRevision: main
    helm:
      parameters:
        - name: instanceName
          value: "fraud-detection"
        - name: dataScienceProjectDisplayName
          value: "Project ${DATA_SCIENCE_PROJECT_NAMESPACE}"
        - name: dataScienceProjectNamespace
          value: "${DATA_SCIENCE_PROJECT_NAMESPACE}"
  syncPolicy:
    automated:
      # prune: true
      selfHeal: true
EOF
