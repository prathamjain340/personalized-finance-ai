# Finance Project API

Standalone FastAPI service for your finance assistant.

## Run Locally

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn finance_project.main:app --host 0.0.0.0 --port 9001
```

Service URLs:
- `http://127.0.0.1:9001/health`
- `http://127.0.0.1:9001/docs`
- `http://127.0.0.1:9001/finance/session/init`

## Docker

Build:

```powershell
docker build -t finance-project-api -f docker/Dockerfile .
```

Run:

```powershell
docker run --rm -p 9001:9001 --env-file .env finance-project-api
```

## Notes For Hosting

- App import path: `finance_project.main:app`
- Start command on most hosts: `python -m uvicorn finance_project.main:app --host 0.0.0.0 --port $PORT`
- Keep this service separate from your older project and call it over HTTP from that project if needed.
