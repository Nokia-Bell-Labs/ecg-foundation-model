# Pretraining Guide

Pretrain CLEF using contrastive learning methods.

## Prerequisites

Ensure the following data files are in place:

| File | File name | Source |
|------|----------|--------|
| Noise data | `DATA_noises_real.mat` | Download from [PhysioNet](https://physionet.org/content/ecg-ppg-simulator-arrhythmia/1.3.1/) |
| Risk scores | `mimiciv_metadata_with_score2.csv` | |
| MIMIC-IV-ECG | - | Download from [PhysioNet](https://physionet.org/content/mimic-iv-ecg/1.0/) |

**Download noise data:**
Note: The noise data must be downloaded from PhysioNet and placed at `dataset/DATA_noises_real.mat` for data augmentation in contrastive pretraining
```bash
wget https://physionet.org/files/ecg-ppg-simulator-arrhythmia/1.3.1/ECG_PPG_model/DATA_noises_real.mat?download -P dataset/
```

Note: The MIMIC‑IV metadata file `mimiciv_metadata_with_score2.csv` will be made publically available.

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
