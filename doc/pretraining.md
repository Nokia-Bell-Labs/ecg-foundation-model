# Pretraining Guide

Pretrain CLEF using contrastive learning methods.

## Prerequisites

Ensure the following data files are in place:

| File | Location | Source |
|------|----------|--------|
| Noise data | `dataset/DATA_noises_real.mat` | Download from [PhysioNet](https://physionet.org/content/ecg-ppg-simulator-arrhythmia/1.3.1/) |

**Download noise data:**
```bash
wget https://physionet.org/files/ecg-ppg-simulator-arrhythmia/1.3.1/ECG_PPG_model/DATA_noises_real.mat?download -P dataset/
```

[TODO]
Note: The MIMIC‑IV metadata file `mimiciv_metadata_with_score2.csv` must be downloaded from Zenodo and placed at
`dataset/mimic-iv/mimiciv_metadata_with_score2.csv` before running pretraining (with metadata).

```bash
wget -O dataset/mimic-iv/mimiciv_metadata_with_score2.csv \
	"https://zenodo.org/record/xxxxxx/files/mimiciv_metadata_with_score2.csv?download=1"
```

---

## Quick Start

```bash
python script/pretrain.py exp.devices=0 exp.pretrain_method=simclr exp.use_metadata=True model.model_size=large
```

**Examples:**
```bash
# CLEF: SimCLR with metadata (proposed method, large model)
python script/pretrain.py exp.devices=0 exp.pretrain_method=simclr exp.use_metadata=True model.model_size=large

# BYOL
python script/pretrain.py exp.devices=0 exp.pretrain_method=byol

# MoCo with single lead
python script/pretrain.py exp.devices=0 exp.pretrain_method=moco
```
