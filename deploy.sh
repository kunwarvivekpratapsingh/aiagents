#!/usr/bin/env bash
# =============================================================================
# deploy.sh — One-command GCP deployment for the Agentic RAG API
#
# What this does:
#   1. Validates prerequisites (gcloud, docker, jq)
#   2. Enables all required GCP APIs
#   3. Creates Artifact Registry repository
#   4. Stores secrets in Secret Manager
#   5. Creates a small VM running the ChromaDB HTTP server
#   6. Builds and pushes the Docker image
#   7. Deploys the FastAPI app to Cloud Run
#   8. Wires Cloud Build trigger for auto-deploy on git push
#   9. Prints the live URL and a test command
#
# Usage:
#   chmod +x deploy.sh
#   ./deploy.sh
#
# Prerequisites:
#   - gcloud CLI installed and authenticated (gcloud auth login)
#   - docker installed and running
#   - jq installed (brew install jq  OR  apt-get install jq)
#   - ANTHROPIC_API_KEY set in your shell environment
# =============================================================================

set -euo pipefail

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'

info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }
step()    { echo -e "\n${BOLD}── $* ──────────────────────────────────────────${NC}"; }

# ── Configuration (override with env vars) ────────────────────────────────────
PROJECT_ID="${GCP_PROJECT_ID:-}"
REGION="${GCP_REGION:-us-central1}"
ZONE="${GCP_ZONE:-${REGION}-a}"
REPO_NAME="${ARTIFACT_REPO:-rag-api}"
SERVICE_NAME="${CLOUD_RUN_SERVICE:-rag-api}"
CHROMA_VM_NAME="${CHROMA_VM:-rag-chromadb}"
CHROMA_VM_TYPE="${CHROMA_VM_TYPE:-e2-small}"
CHROMA_DISK_SIZE="${CHROMA_DISK_SIZE:-50GB}"
GITHUB_ORG="${GITHUB_ORG:-}"
GITHUB_REPO="${GITHUB_REPO:-aiagents}"

# ── Step 0: Prerequisites ─────────────────────────────────────────────────────
step "Checking prerequisites"

for cmd in gcloud docker jq; do
  command -v "$cmd" &>/dev/null || error "$cmd is required. Install it first."
done

gcloud auth print-access-token &>/dev/null || \
  error "Not authenticated. Run: gcloud auth login"

gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet 2>/dev/null || true

# Resolve project ID
if [[ -z "$PROJECT_ID" ]]; then
  PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
  [[ -z "$PROJECT_ID" ]] && \
    error "No project set. Run: gcloud config set project YOUR_PROJECT_ID\n       or: export GCP_PROJECT_ID=your-project-id"
fi

PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/agentic-rag"

success "Project: ${PROJECT_ID} (${PROJECT_NUMBER})"
success "Region:  ${REGION}"
success "Image:   ${IMAGE}"

# ── Step 1: Enable APIs ───────────────────────────────────────────────────────
step "Enabling GCP APIs"

gcloud services enable \
  run.googleapis.com \
  compute.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com \
  cloudbuild.googleapis.com \
  cloudresourcemanager.googleapis.com \
  --project="$PROJECT_ID" --quiet

success "APIs enabled"

# ── Step 2: Artifact Registry ─────────────────────────────────────────────────
step "Setting up Artifact Registry"

if ! gcloud artifacts repositories describe "$REPO_NAME" \
     --location="$REGION" --project="$PROJECT_ID" &>/dev/null; then
  gcloud artifacts repositories create "$REPO_NAME" \
    --repository-format=docker \
    --location="$REGION" \
    --project="$PROJECT_ID" \
    --description="Agentic RAG Docker images"
  success "Created repository: ${REPO_NAME}"
else
  info "Repository ${REPO_NAME} already exists"
fi

# ── Step 3: Secrets ───────────────────────────────────────────────────────────
step "Storing secrets in Secret Manager"

# Anthropic API key
if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  read -rsp "Enter your Anthropic API key (sk-ant-...): " ANTHROPIC_API_KEY
  echo
fi

if ! gcloud secrets describe anthropic-api-key --project="$PROJECT_ID" &>/dev/null; then
  printf '%s' "$ANTHROPIC_API_KEY" | \
    gcloud secrets create anthropic-api-key \
      --data-file=- --project="$PROJECT_ID"
  success "Created secret: anthropic-api-key"
else
  printf '%s' "$ANTHROPIC_API_KEY" | \
    gcloud secrets versions add anthropic-api-key \
      --data-file=- --project="$PROJECT_ID"
  info "Updated secret: anthropic-api-key"
fi

# RAG API key (used by callers of your API)
if ! gcloud secrets describe rag-api-key --project="$PROJECT_ID" &>/dev/null; then
  RAG_API_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
  printf '%s' "$RAG_API_KEY" | \
    gcloud secrets create rag-api-key \
      --data-file=- --project="$PROJECT_ID"
  success "Generated and stored API key. Save this — you'll need it for all API calls:"
  echo -e "  ${BOLD}${GREEN}API_KEY=${RAG_API_KEY}${NC}"
else
  info "Secret rag-api-key already exists (not rotated)"
  RAG_API_KEY=$(gcloud secrets versions access latest \
    --secret=rag-api-key --project="$PROJECT_ID")
fi

# ── Step 4: ChromaDB VM ────────────────────────────────────────────────────────
step "Creating ChromaDB VM"

if ! gcloud compute instances describe "$CHROMA_VM_NAME" \
     --zone="$ZONE" --project="$PROJECT_ID" &>/dev/null; then

  gcloud compute instances create "$CHROMA_VM_NAME" \
    --machine-type="$CHROMA_VM_TYPE" \
    --image-family=debian-12 \
    --image-project=debian-cloud \
    --boot-disk-size="$CHROMA_DISK_SIZE" \
    --boot-disk-type=pd-ssd \
    --zone="$ZONE" \
    --project="$PROJECT_ID" \
    --no-address \
    --metadata=startup-script='#!/bin/bash
      set -e
      apt-get update -q
      curl -fsSL https://get.docker.com | sh
      mkdir -p /data/chroma
      docker run -d --restart=always --name chromadb \
        -p 8001:8000 \
        -v /data/chroma:/chroma/chroma \
        -e IS_PERSISTENT=TRUE \
        -e ANONYMIZED_TELEMETRY=FALSE \
        chromadb/chroma:latest
      echo "ChromaDB started" > /tmp/chromadb-ready
    '

  success "Created VM: ${CHROMA_VM_NAME}"
  info "Waiting 60s for VM startup script to complete..."
  sleep 60
else
  info "VM ${CHROMA_VM_NAME} already exists"
fi

CHROMA_INTERNAL_IP=$(gcloud compute instances describe "$CHROMA_VM_NAME" \
  --zone="$ZONE" --project="$PROJECT_ID" \
  --format='value(networkInterfaces[0].networkIP)')

success "ChromaDB internal IP: ${CHROMA_INTERNAL_IP}"

# Firewall rule: allow Cloud Run to reach ChromaDB on port 8001
if ! gcloud compute firewall-rules describe allow-chromadb-internal \
     --project="$PROJECT_ID" &>/dev/null; then
  gcloud compute firewall-rules create allow-chromadb-internal \
    --direction=INGRESS \
    --action=ALLOW \
    --rules=tcp:8001 \
    --source-ranges=0.0.0.0/0 \
    --target-tags=chromadb-server \
    --project="$PROJECT_ID"

  gcloud compute instances add-tags "$CHROMA_VM_NAME" \
    --tags=chromadb-server \
    --zone="$ZONE" \
    --project="$PROJECT_ID"
fi

# ── Step 5: Grant Cloud Build permissions ─────────────────────────────────────
step "Setting IAM permissions for Cloud Build"

CB_SA="${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com"
CR_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

for role in roles/run.admin roles/iam.serviceAccountUser \
            roles/secretmanager.secretAccessor roles/artifactregistry.writer; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${CB_SA}" \
    --role="$role" \
    --condition=None --quiet 2>/dev/null || true
done

# Cloud Run runtime SA needs to read secrets
gcloud secrets add-iam-policy-binding anthropic-api-key \
  --member="serviceAccount:${CR_SA}" \
  --role=roles/secretmanager.secretAccessor \
  --project="$PROJECT_ID" --quiet 2>/dev/null || true

gcloud secrets add-iam-policy-binding rag-api-key \
  --member="serviceAccount:${CR_SA}" \
  --role=roles/secretmanager.secretAccessor \
  --project="$PROJECT_ID" --quiet 2>/dev/null || true

success "IAM permissions set"

# ── Step 6: Build and push Docker image ───────────────────────────────────────
step "Building and pushing Docker image"

SHORT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "manual")

docker build \
  -t "${IMAGE}:${SHORT_SHA}" \
  -t "${IMAGE}:latest" \
  .

docker push "${IMAGE}:${SHORT_SHA}"
docker push "${IMAGE}:latest"

success "Pushed: ${IMAGE}:${SHORT_SHA}"

# ── Step 7: Deploy to Cloud Run ───────────────────────────────────────────────
step "Deploying to Cloud Run"

gcloud run deploy "$SERVICE_NAME" \
  --image="${IMAGE}:${SHORT_SHA}" \
  --region="$REGION" \
  --project="$PROJECT_ID" \
  --platform=managed \
  --execution-environment=gen2 \
  --memory=2Gi \
  --cpu=2 \
  --min-instances=1 \
  --max-instances=10 \
  --concurrency=80 \
  --timeout=600 \
  --no-allow-unauthenticated \
  --set-secrets="ANTHROPIC_API_KEY=anthropic-api-key:latest,API_KEY=rag-api-key:latest" \
  --set-env-vars="CHROMA_HOST=${CHROMA_INTERNAL_IP},CHROMA_PORT=8001,JSON_LOGS=true,LOG_LEVEL=INFO,WORKERS=1" \
  --quiet

SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" \
  --region="$REGION" --project="$PROJECT_ID" \
  --format='value(status.url)')

success "Deployed: ${SERVICE_URL}"

# ── Step 8: Cloud Build trigger (optional) ────────────────────────────────────
step "Setting up Cloud Build trigger"

if [[ -n "$GITHUB_ORG" ]]; then
  # Check if trigger already exists
  if ! gcloud builds triggers describe "${SERVICE_NAME}-trigger" \
       --project="$PROJECT_ID" &>/dev/null 2>&1; then
    gcloud builds triggers create github \
      --name="${SERVICE_NAME}-trigger" \
      --repo-owner="$GITHUB_ORG" \
      --repo-name="$GITHUB_REPO" \
      --branch-pattern='^main$' \
      --build-config=cloudbuild.yaml \
      --substitutions="_CHROMA_HOST=${CHROMA_INTERNAL_IP},_REGION=${REGION},_REPO=${REPO_NAME},_SERVICE=${SERVICE_NAME}" \
      --project="$PROJECT_ID"
    success "Cloud Build trigger created — auto-deploys on push to main"
  else
    info "Cloud Build trigger already exists"
  fi
else
  warn "Skipping Cloud Build trigger (set GITHUB_ORG=your-org to enable auto-deploy)"
fi

# ── Step 9: Smoke test ────────────────────────────────────────────────────────
step "Smoke testing deployment"

HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "${SERVICE_URL}/health")
if [[ "$HTTP_STATUS" == "200" ]]; then
  success "Health check passed (HTTP ${HTTP_STATUS})"
else
  warn "Health check returned HTTP ${HTTP_STATUS} — check Cloud Run logs"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}════════════════════════════════════════════════${NC}"
echo -e "${BOLD}${GREEN}  Deployment complete!${NC}"
echo -e "${BOLD}${GREEN}════════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${BOLD}API URL:${NC}  ${SERVICE_URL}"
echo -e "  ${BOLD}API Key:${NC}  ${RAG_API_KEY}"
echo ""
echo -e "${BOLD}Test it:${NC}"
echo "  curl ${SERVICE_URL}/health"
echo ""
echo "  curl -X POST ${SERVICE_URL}/chat \\"
echo "    -H 'X-API-Key: ${RAG_API_KEY}' \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"query\": \"What is RAG?\"}'"
echo ""
echo -e "${BOLD}View logs:${NC}"
echo "  gcloud logging tail 'resource.type=cloud_run_revision AND resource.labels.service_name=${SERVICE_NAME}' --project=${PROJECT_ID} --format=json"
echo ""
echo -e "${BOLD}Upload a document:${NC}"
echo "  curl -X POST ${SERVICE_URL}/documents/upload \\"
echo "    -H 'X-API-Key: ${RAG_API_KEY}' \\"
echo "    -F 'file=@yourfile.pdf'"
echo ""
