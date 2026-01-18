# reel-driver-modeling

ML training components for the Reel Driver media recommendation system.

## Overview

This project provides containerized ML training components for the Reel Driver recommendation system. It includes feature engineering and hyperparameter optimization modules that run as stateless containers in a Dagster pipeline.

For the inference API service, see the [reel-driver](https://github.com/your-org/reel-driver) repository.

## Project Structure

```
reel-driver-modeling/
├── src/                               # Training pipeline source code
│   ├── training/
│   │   ├── feature_engineering.py    # Feature engineering pipeline
│   │   └── model_training.py         # ML model training & tuning
│   └── utils/
│       └── db_operations.py          # Database utilities
├── tests/                            # Test suite
│   ├── training/                    # Training pipeline tests
│   │   ├── test_feature_engineering_integration.py
│   │   ├── test_model_training_integration.py
│   │   └── test_model_training_unit.py
│   └── conftest.py                  # Shared test fixtures
├── containerization/                # Docker container definitions
│   ├── dockerfile.base_training     # CPU base training image
│   ├── dockerfile.base_training_gpu # GPU base training image (CUDA 12.4)
│   ├── dockerfile.feature_engineering_cpu
│   ├── dockerfile.feature_engineering_gpu
│   ├── dockerfile.model_training_cpu
│   └── dockerfile.model_training_gpu
├── notebooks/                       # Jupyter notebooks for analysis
├── config/                          # Configuration files
├── pyproject.toml                   # Project configuration and dependencies
├── uv.lock                         # Locked dependencies for reproducible builds
└── CLAUDE.md                       # Detailed technical documentation
```

## Quick Start

### Prerequisites

- Python 3.12+
- PostgreSQL database (for training data)
- MLflow server (for model tracking and artifacts)
- MinIO or S3-compatible storage (for MLflow artifacts)

### Environment Variables

Create a `.env` file in the root directory:

```bash
# MLflow Configuration
REEL_DRIVER_MLFLOW_HOST=mlflow-host
REEL_DRIVER_MLFLOW_PORT=5000
REEL_DRIVER_MLFLOW_EXPERIMENT=reel-driver-experiment
REEL_DRIVER_MLFLOW_MODEL=reel-driver-model

# MinIO Configuration (for MLflow artifacts)
REEL_DRIVER_MINIO_ENDPOINT=http://minio-host
REEL_DRIVER_MINIO_PORT=9000
REEL_DRIVER_MINIO_ACCESS_KEY=your-access-key
REEL_DRIVER_MINIO_SECRET_KEY=your-secret-key

# Database Configuration (for training)
REEL_DRIVER_TRNG_PGSQL_HOST=postgresql-host
REEL_DRIVER_TRNG_PGSQL_PORT=5432
REEL_DRIVER_TRNG_PGSQL_DATABASE=database-name
REEL_DRIVER_TRNG_PGSQL_SCHEMA=schema-name
REEL_DRIVER_TRNG_PGSQL_USERNAME=username
REEL_DRIVER_TRNG_PGSQL_PASSWORD=password

# Development Mode
LOCAL_DEVELOPMENT=true  # Set to 'true' for local development
```

### Installation

This project uses `uv` for dependency management with modular dependencies:

```bash
# Install dependencies based on your needs:

# For feature engineering
uv sync --extra feature-engineering

# For model training
uv sync --extra model-training

# For running tests (includes dev dependencies)
uv sync

# For development with all extras
uv sync --all-extras

# Note: uv automatically creates and manages the virtual environment in .venv
# To activate the environment manually: source .venv/bin/activate
```

## Testing

### Run All Tests
```bash
# Run complete test suite
uv run pytest tests/training/

# Run with verbose output
uv run pytest tests/training/ -v
```

### Run Specific Test Categories
```bash
# Unit tests only (fast, mocked dependencies)
uv run pytest tests/training/test_model_training_unit.py -v

# Integration tests
uv run pytest tests/training/test_model_training_integration.py -v
uv run pytest tests/training/test_feature_engineering_integration.py -v
```

### Integration Tests
Integration tests require real services. Set `LOCAL_DEVELOPMENT=true` in your `.env`:

```bash
uv run pytest tests/training/ -v
```

## Training Pipeline

### Feature Engineering
```bash
uv run python -c "from src.training.feature_engineering import __main__; __main__()"
```

### Model Training
```bash
uv run python -c "from src.training.model_training import __main__; __main__()"
```

## Container Deployment

### Build Images
```bash
# Build CPU training containers
docker build -f containerization/dockerfile.base_training -t reel-driver-training-base-cpu .
docker build -f containerization/dockerfile.feature_engineering_cpu -t reel-driver-feature-engineering-cpu .
docker build -f containerization/dockerfile.model_training_cpu -t reel-driver-model-training-cpu .

# Build GPU training containers (requires NVIDIA Docker runtime)
docker build -f containerization/dockerfile.base_training_gpu -t reel-driver-training-base-gpu .
docker build -f containerization/dockerfile.feature_engineering_gpu -t reel-driver-feature-engineering-gpu .
docker build -f containerization/dockerfile.model_training_gpu -t reel-driver-model-training-gpu .
```

### Run Containers
```bash
# Run CPU training containers
docker run --env-file .env reel-driver-feature-engineering-cpu
docker run --env-file .env reel-driver-model-training-cpu

# Run GPU training containers (requires nvidia-docker)
docker run --gpus all --env-file .env reel-driver-feature-engineering-gpu
docker run --gpus all --env-file .env reel-driver-model-training-gpu

# Or use docker-compose for local GPU/CPU training
docker-compose -f docker-compose.training.yaml --profile gpu up model-training-gpu
docker-compose -f docker-compose.training.yaml --profile cpu up model-training-cpu
```

## Documentation

- **[CLAUDE.md](./CLAUDE.md)** - Comprehensive technical documentation
- **[tests/README.md](./tests/README.md)** - Testing documentation
- **Jupyter Notebooks** - In the `notebooks/` directory for data analysis

## Architecture

This project implements the training layer of an MLOps pipeline:

1. **Stateless Training Containers** - Feature engineering and model training with CPU/GPU variants
2. **GPU Acceleration** - XGBoost training with CUDA support (22-25% speedup on current dataset)
3. **MLflow Integration** - Experiment tracking and model registry
4. **Comprehensive Testing** - Unit tests, integration tests, and CI/CD
5. **Container Orchestration** - Designed for Dagster pipeline orchestration

### Docker Images

| Image | Description |
|-------|-------------|
| `reel-driver-training-base-cpu` | CPU base image (Python 3.12) |
| `reel-driver-training-base-gpu` | GPU base image (CUDA 12.4) |
| `reel-driver-feature-engineering-cpu/gpu` | Feature engineering pipeline |
| `reel-driver-model-training-cpu/gpu` | XGBoost model training with Optuna |

See [docs/CPU-v-GPU.md](./docs/CPU-v-GPU.md) for detailed GPU implementation documentation.
