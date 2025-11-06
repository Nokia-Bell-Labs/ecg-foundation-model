# Environment Setup

This guide covers installation options for CLEF.

## Prerequisites

- Linux system (tested on Ubuntu 20.04)
- NVIDIA GPU with CUDA support (recommended)
- Docker (for containerized setup) or Conda

---

## Option 1: Docker (Recommended)

Docker provides a consistent environment across systems.

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
  -v /path/to/datasets:/datasets \
  -v /path/to/models:/models \
  clef_image
```

**Volume mounts (customize as needed):**
- `-v $(pwd):/CLEF` — mount code directory
- `-v /path/to/datasets:/datasets` — mount dataset directory
- `-v /path/to/models:/models` — mount model checkpoint directory

### 3. Access the Container

```bash
docker exec -it clef_container /bin/bash
```

### 4. Install CLEF Package

Inside the container:

```bash
cd /CLEF
pip install -e .
```

---

## Option 2: Conda Environment

If you prefer a local conda setup:

### 1. Create Conda Environment

```bash
conda create -n clef python=3.12
conda activate clef
```

### 2. Install Dependencies

```bash
cd /path/to/CLEF
pip install -r requirements.txt
pip install -e .
```

### 3. Verify Installation

```bash
python -c "import clef; print(clef.__version__)"
```

---

## Verify GPU Access

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

Expected output: `CUDA available: True`

---

## Troubleshooting

### Docker: Permission Denied

```bash
sudo usermod -aG docker $USER
newgrp docker
```

### Conda: CUDA Mismatch

Ensure PyTorch CUDA version matches your system:

```bash
python -c "import torch; print(torch.version.cuda)"
nvidia-smi
```

Reinstall PyTorch with correct CUDA version if needed:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## Next Steps

- [Dataset Preparation](dataset_preparation.md)
- [Download Pretrained Weights](get_pretrained_weight.md)
- [Run Pretraining](pretraining.md)