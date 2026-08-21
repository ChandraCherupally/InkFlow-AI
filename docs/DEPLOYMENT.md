# InkFlow-AI — Deployment & CI/CD Guide

Welcome to the **InkFlow-AI** Deployment Guide! This guide explains how to run InkFlow-AI locally, in containerized Docker environments, and through our automated **AWS & GitHub Actions CI/CD pipeline**.

---

## 📋 Table of Contents
1. [Prerequisites & Environment Setup](#1-prerequisites--environment-setup)
2. [Local Development Setup](#2-local-development-setup)
3. [Containerized Setup with Docker](#3-containerized-setup-with-docker)
4. [Production CI/CD Pipeline (AWS & GitHub Actions)](#4-production-cicd-pipeline-aws--github-actions)
5. [Twelve-Factor Security & Secret Management](#5-twelve-factor-security--secret-management)
6. [Troubleshooting & Health Checks](#6-troubleshooting--health-checks)

---

## 1. Prerequisites & Environment Setup

Before running or deploying InkFlow-AI, ensure you have API keys for the required providers.

### Environment Variables (`.env`)
Create a `.env` file in the root directory:

```env
# AI Provider Keys (Required)
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here

# Google Cloud / Vertex AI Credentials (Optional for local, required for Vertex AI)
PROJECT_ID=your_gcp_project_id
REGION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=/app/vertex-key.json

# LangSmith Observability (Optional)
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=InkFlow-AI
```

---

## 2. Local Development Setup

### Option A: Using `uv` (Fastest & Recommended)
```bash
# 1. Clone the repository
git clone https://github.com/ChandraCherupally/InkFlow-AI.git
cd InkFlow-AI

# 2. Create virtual environment and install dependencies
uv venv
uv sync

# 3. Start the FastAPI server
uv run uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Option B: Standard Python `pip`
```bash
# 1. Create venv
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run server
uvicorn app:app --reload --port 8000
```

Access the UI in your browser at `http://localhost:8000`.

---

## 3. Containerized Setup with Docker

InkFlow-AI includes ready-to-use Docker configurations located in the `docker/` folder.

### Using Docker Compose (Recommended)
To start the application container with persistent volume mounts for output files:

```bash
# Build and run the container in detached mode
docker compose -f docker/docker-compose.yml up -d --build
```

- **App URL**: `http://localhost:8000`
- **Output Storage**: Local `./outputs` directory is bind-mounted into `/app/outputs` in the container.

To stop the service:
```bash
docker compose -f docker/docker-compose.yml down
```

---

## 4. Production CI/CD Pipeline (AWS & GitHub Actions)

InkFlow-AI uses a **sequential, automated CI/CD pipeline** with GitHub Actions and Docker Hub to deploy to an AWS EC2 instance.

```
 [ Developer Push to main ]
             │
             ▼
 ┌───────────────────────┐
 │   CI Workflow (ci.yml)│  ← Ubuntu runner: pytest + Docker build test
 └───────────┬───────────┘
             │
             │ (On CI Success)
             ▼
 ┌───────────────────────┐
 │   CD Workflow (cd.yml)│  ← AWS EC2 Self-Hosted Runner
 └───────────┬───────────┘
             │
             ├──► 1. Build & Push Docker image to Docker Hub (latest + SHA tags)
             ├──► 2. Generate runtime .env & restricted vertex-key.json (chmod 600)
             ├──► 3. Pull latest image & execute `docker compose up -d`
             └──► 4. Run automated HTTP health checks (/api/health)
```

### Workflow Details

#### 1. Continuous Integration (`.github/workflows/ci.yml`)
- Triggered on: Every `push` to `main` and all `pull_request`s.
- Steps:
  1. Sets up Python 3.12 and `uv`.
  2. Runs pytest test suite (`uv run pytest tests/ -v`). Fails build if any test fails.
  3. Builds Docker image using BuildKit layer caching (no push).

#### 2. Continuous Deployment (`.github/workflows/cd.yml`)
- Triggered on: `workflow_run` (only after `CI` workflow finishes with `success`).
- Executed on: **Self-hosted runner** running directly on the **AWS EC2 instance**.
- Steps:
  1. Authenticates with Docker Hub.
  2. Builds & pushes multi-tagged Docker images (`latest` and `github.sha`).
  3. Prepares deployment directory `~/inkflow-ai/`.
  4. Generates Vertex AI credentials (`vertex-key.json`) securely on host.
  5. Generates production `.env` from GitHub Secrets.
  6. Restarts containers using `docker compose up -d`.
  7. Cleans old dangling images (`docker image prune -f`).
  8. Validates `/api/health` endpoint.

---

## 5. Twelve-Factor Security & Secret Management

InkFlow-AI adheres strictly to Twelve-Factor app security principles:

1. **Zero Hardcoded Secrets**: Secrets and keys are never stored in code, Dockerfiles, or committed to Git.
2. **Runtime Secret Injection**:
   - Secrets are managed centrally via **GitHub Repository Secrets**.
   - During CD deployment, secrets are injected into `~/inkflow-ai/.env` on the EC2 host.
3. **Restricted Credential File Permissions**:
   - `VERTEX_SERVICE_ACCOUNT_JSON` is written at deploy time to `~/inkflow-ai/vertex-key.json`.
   - Permissions are locked to `600` (read/write by owner only).
   - Mounted read-only (`:ro`) into the container at `/app/vertex-key.json`.

---

## 6. Troubleshooting & Health Checks

### Check Container Status
```bash
docker compose -f docker/docker-compose.yml ps
```

### View Application Logs
```bash
docker compose -f docker/docker-compose.yml logs -f
```

### Direct Health Check Endpoint
```bash
curl http://localhost:8000/api/health
```

Expected JSON response:
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```
