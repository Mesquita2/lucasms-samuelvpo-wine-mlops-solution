set windows-shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-Command"]

# Lista as receitas disponíveis
default:
    @just --list

# Sincroniza dependências
install:
    uv sync

# Inicia o MLflow server local (porta 5000)
mlflow:
    uv run mlflow server --host 127.0.0.1 --port 5000 \
        --backend-store-uri sqlite:///mlflow.db

# Cria o champion inicial (executar uma única vez após subir o MLflow)
bootstrap:
    uv run python -m scripts.bootstrap_champion

# Treina um challenger
train n_estimators="200" max_depth="8":
    uv run python -m src.train --n_estimators {{n_estimators}} --max_depth {{max_depth}}

# Avalia um run no conjunto de teste
evaluate run_id:
    uv run python -m src.evaluate --run-id {{run_id}}

# Executa o teste A/B e promove se challenger vencer
promote:
    uv run python -m src.promote

# Pipeline completo: treina + avalia o último run + promove
pipeline:
    uv run python -m src.train --n_estimators 200 --max_depth 8
    uv run python -c "import mlflow; mlflow.set_tracking_uri('http://127.0.0.1:5000'); \
        rid = mlflow.search_runs(experiment_names=['wine-classifier'], \
        order_by=['attributes.start_time DESC']).iloc[0]['run_id']; \
        import subprocess; subprocess.check_call(['uv','run','python','-m','src.evaluate','--run-id',rid])"
    uv run python -m src.promote

# Sobe a API local
serve:
    uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Testes
test:
    uv run pytest

# Lint
lint:
    uv run ruff check .

# Build da imagem Docker
docker-build:
    docker build -t wine-classifier:latest .

# Sobe tudo com docker compose
up:
    docker compose up -d

down:
    docker compose down
