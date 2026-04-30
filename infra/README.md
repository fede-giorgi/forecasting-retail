# Deployment — n8n + Forecast API

Two services:

- **`forecast-api`** (port 8000): FastAPI exposing `POST /forecast` for the agent.
- **`n8n`** (port 5678): visual workflow builder + AI Agent runtime.

`forecast-api` reads `data/processed/agent_*.parquet` (mounted read-only). Run the playground first to generate them.

## Local (laptop)

```bash
# 0. Run the playground once to produce data/processed/agent_*.parquet
# 1. Set secrets
echo "GEMINI_API_KEY=..." >> .env          # or OPENAI_API_KEY for n8n's AI Agent
echo "N8N_USER=admin"     >> .env
echo "N8N_PASSWORD=..."   >> .env

# 2. Boot
cd infra
docker compose up -d --build

# 3. Verify
curl http://localhost:8000/health
open http://localhost:5678                  # log in with the credentials above

# 4. Import the workflow
# In n8n UI → Workflows → Import from File → select agent/n8n_workflow_example.json
# In the workflow, edit "Forecast Lookup Tool" URL to: http://forecast-api:8000/forecast
# (NOT host.docker.internal — that's only for docker-host access; here both are in compose)
```

If the n8n AI-Agent node needs an LLM, the easiest path is OpenAI (set `OPENAI_API_KEY`); n8n also supports Gemini natively.

## On AWS (~$15/month, fits in your $100 credit)

```bash
# 1. Launch EC2 t3.small (Amazon Linux 2023), open ports 5678 + 8000
# 2. Install Docker + git
sudo dnf install -y docker git
sudo systemctl enable --now docker
sudo usermod -aG docker ec2-user && exec sudo su - ec2-user

# 3. Clone + build
git clone https://github.com/<user>/forecasting-retail
cd forecasting-retail
# Either: scp data/processed/*.parquet up to EC2, OR rerun the playground there
cd infra
echo "GEMINI_API_KEY=..." > .env
echo "OPENAI_API_KEY=..." >> .env
echo "N8N_USER=admin"     >> .env
echo "N8N_PASSWORD=..."   >> .env
docker compose up -d --build
```

For production, put both behind nginx + Let's Encrypt and add a bearer token on `/forecast`.

## Switching to n8n Cloud (no Docker needed)

If you'd rather skip self-hosting:

1. Sign up at https://n8n.io (free 14-day trial, then ~$20/month).
2. Run **only** `forecast-api` somewhere reachable from the Internet (EC2 + ngrok works for testing).
3. In the n8n Cloud workflow, set the HTTP-Tool URL to your public endpoint.
4. Add a bearer token on the API and set it as a header in n8n.
