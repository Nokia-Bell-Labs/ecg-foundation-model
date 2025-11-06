# Downstream Tasks Guide

Evaluate pretrained models on downstream ECG tasks.

## Quick Start

```bash
python script/downstream.py model="<MODEL>" dataset="<DATASET>" exp="<EXP>" task.type="<TASK_TYPE>" [exp.task="<TASK>"]
```

**Examples:**
```bash
# CLEF (our pretrained model)
python script/downstream.py model="clef" model.model_size=large model.statekey_file="clef-large.ckpt" dataset="mcmed_data" exp="mcmed_downstream" task.type="classification" exp.task="ed-dispo"

# Baseline: KED
python script/downstream.py model="ked" dataset="music_data" exp="music_downstream" task.type="classification"

# Baseline: Regression with task selection
python script/downstream.py model="ked" dataset="aurorabp_data" exp="aurorabp_downstream" task.type="regression" exp.task="sbp"

# Baseline: PTB-XL with specific task
python script/downstream.py model="ecgfounder" dataset="ptbxl_data" exp="ptbxl_downstream" task.type="classification" exp.task="diagnostic"
```

---

## Using CLEF (Our Pretrained Model)

```bash
python script/downstream.py \
    model="clef" \
    model.model_size="<SIZE>" \
    model.statekey_file="clef-<SIZE>.ckpt" \
    dataset="<DATASET>" \
    exp="<EXP>" \
    task.type="<TASK_TYPE>" \
    [exp.task="<TASK>"]
```

**Model sizes:** `small`, `medium`, `large`

**Example with all parameters:**
```bash
python script/downstream.py \
    model="clef" \
    model.model_size="large" \
    model.statekey_file="clef-large.ckpt" \
    dataset="mcmed_data" \
    exp="mcmed_downstream" \
    task.type="classification" \
    exp.task="ed-dispo" \
    exp.devices=0 \
    model.model_params.linear_prob=false
```

---

## Supported Baseline Models

| Model | Config |
|-------|--------|
| `clef` | **CLEF (our pretrained model)** |
| `ked` | KED (ECG-FM) |
| `ecgfounder` | ECGFounder |
| `stmem` | ST-MEM |
| `moment` | MOMENT |
| `moirai` | Moirai |

---

## Supported Tasks

### Classification

| Dataset | Config | Exp | Tasks |
|---------|--------|-----|-------|
| MUSIC | `music_data` | `music_downstream` | (default) |
| Chapman | `chapman_data` | `chapman_downstream` | (default) |
| MIMIC-IV | `mimiciv_data` | `mimiciv_downstream` | (default) |
| PTB-XL | `ptbxl_data` | `ptbxl_downstream` | `diagnostic`, `subdiagnostic`, `superdiagnostic`, `form`, `rhythm` |
| Icentia11K | `icentia_data` | `icentia_downstream` | `beat`, `rhythm` |
| MCMED | `mcmed_data` | `mcmed_downstream` | `ed-dispo`, `dc-dispo`, `acuity` |

### Regression

| Dataset | Config | Exp | Tasks |
|---------|--------|-----|-------|
| MIMIC-IV | `mimiciv_data` | `mimiciv_downstream` | (default) |
| MCMED | `mcmed_data` | `mcmed_downstream` | `sbp`, `dbp` |
| AuroraBP | `aurorabp_data` | `aurorabp_downstream` | `sbp`, `dbp` |

---

## Dataset Configuration

Edit `configs/dataset/<dataset>_data.yaml` and set `data_path` to the directory containing:

| Dataset | Required File/Folder in `data_path` |
|---------|--------------------------------------|
| **AuroraBP** | folder containing `participants.tsv` file |
| **Icentia11K** | folder containing `RECORDS` file |
| **MCMED** | folder containing `split_chrono_train.csv` file |
| **MIMIC-IV** | point to `mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/` |
| **MUSIC** | point to `Holter_ECG/` folder |
| **PTB-XL** | point to `PTBXL/1.0.3/` directory |
| **Chapman** | `dataset/chapman` (directory with processed data, see `dataset_preparation.md`) |


## Run All Experiments

To run all the experiments, run:

```bash
bash script/run_baselines.sh
```

This runs all 5 baseline models on all tasks.

