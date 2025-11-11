# Environment Setup

This guide covers installation options for CLEF.

## Prerequisites

- Linux system (tested on Ubuntu 20.04)
- NVIDIA GPU with CUDA support (recommended)
- Conda or uv for local setup, Docker for containerized environment

---

## Option 1: Conda Environment

Standard setup using conda:

### 1. Create Conda Environment

```bash
conda create -n clef python=3.12
conda activate clef
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install CLEF Package

```bash
pip install -e .
```

---

## Option 2: uv (Fast Setup)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver.

### 1. Sync Dependencies

From the project root:

```bash
uv sync
```

This will create a virtual environment in `.venv` and install all dependencies from `pyproject.toml`.

### 2. Activate Virtual Environment

```bash
source .venv/bin/activate
```

### 3. Install Additional Dependencies

Install fastai (requires pip as it's not compatible with uv):

```bash
uv pip install fastai
```

### 4. Install CLEF Package in Editable Mode

```bash
uv run python -m pip install -e .
```

Or, with the virtual environment activated:

```bash
pip install -e .
```

---

## Option 3: Docker (For Reproducibility)

Docker provides a consistent, reproducible environment across different systems.

### 1. Build the Docker Image

```bash
cd /path/to/CLEF
docker build -t clef_image .
```

### 2. Run the Container

```bash
docker run -dit \
  --gpus all \
  --name clef_container \
  --shm-size=8G \
  -v $(pwd):/CLEF \
  clef_image
```

### 3. Access the Container

```bash
docker exec -it clef_container /bin/bash
```

### 4. Install Dependencies and CLEF Package

Inside the container, follow the same installation steps as Option 1 (Conda) or Option 2 (uv).

---

## Verify GPU Access

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

Expected output: `CUDA available: True`





## Next Steps

- [Dataset Preparation](dataset_preparation.md)
- [Download Pretrained Weights](get_pretrained_weight.md)
- [Run Pretraining](pretraining.md)