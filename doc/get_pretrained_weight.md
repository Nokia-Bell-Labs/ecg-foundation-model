# Pretrained Weight Setup Guide

## Manual Download Required

### CLEF 
Model weights of our pretrained models will be made publicallt available.

### ST-MEM
- **Download**: [Google Drive](https://drive.google.com/file/d/14nScwPk35sFi8wc-cuLJLqudVwynKS0n/view?usp=share_link)
- **Paper**: https://openreview.net/pdf?id=WcOohbsF4H
- **Repo**: https://github.com/bakqui/ST-MEM

### KED (ECGFM-KED)
- **Download**: [Zenodo](https://zenodo.org/records/14881564)
- **Paper**: https://doi.org/10.1016/j.xcrm.2024.101875
- **Repo**: https://github.com/control-spiderman/ECGFM-KED

### ECGFounder
- **Download**: [Hugging Face](https://huggingface.co/PKUDigitalHealth/ECGFounder/tree/main)
- **Paper**: https://arxiv.org/abs/2410.04133
- **Repo**: https://github.com/PKUDigitalHealth/ECGFounder


## Automatic Download (No Action Needed)

### Moirai
- **Paper**: https://arxiv.org/abs/2402.02592
- **Repo**: https://github.com/SalesforceAIResearch/uni2ts
- **Model**: https://huggingface.co/Salesforce/moirai-1.1-R-base

### Moment
- **Paper**: https://arxiv.org/abs/2402.03885
- **Repo**: https://github.com/moment-timeseries-foundation-model/moment
- **Model**: https://huggingface.co/AutonLab/MOMENT-1-base

## Expected Directory Structure

After downloading, your `models/` directory should look like:

```
models/
├── ecgfm-ked/
│   └── best_valid_all_increase_with_augment_epoch_3.pt
├── ecgfounder/
│   ├── 1_lead_ECGFounder.pth
│   └── 12_lead_ECGFounder.pth
└── st-mem/
    ├── st_mem_vit_base_encoder.pth
    └── st_mem_vit_base_full.pth
```

**Note**: 
- Moirai and Moment weights are automatically downloaded from Hugging Face on first use
- Checkpoint paths are configured via Hydra config files (see `configs/model/*.yaml`)
- You can override paths using: `python script/downstream.py model.checkpoint_path="/custom/path"`