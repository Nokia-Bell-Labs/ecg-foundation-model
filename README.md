<div align="center">
  <table>
    <tr>
      <td><img src="figures/clef_logo.png" alt="CLEF Logo" width="100"></td>
      <td>
        <h1>CLEF</h1>
        <h2>Clinically-Guided Contrastive Learning for<br>Electrocardiogram Foundation Models</h2>
      </td>
    </tr>
  </table>
  
  <p>
    <a href="https://arxiv.org/abs/xxxxx"><img src="https://img.shields.io/badge/arXiv-2410.20542-b31b1b.svg" alt="ArXiv"></a>
    <a href="https://zenodo.org/records/xxxx"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.13983110.svg" alt="DOI"></a>
  </p>
</div>

## 🌟 Overview

The electrocardiogram~(ECG) is a key diagnostic tool in cardiovascular health. Single-lead ECG recording is integrated into both clinical-grade and consumer wearables. We propose CLEF, the first **foundation model for single-lead ECG**, leveraging metadata–derived risk scores for each patient as guided supervisory signals. CLEF was pretrained on 161K patients from MIMIC-IV-ECG using 12-lead ECGs. We evaluated on **18 clinical classification and regression tasks** across **7 held-out datasets**, and benchmarked against 5 foundation model baselines and 3 self-supervised learning algorithms. Overall, out method achieves an **≥ 2.6% improvement in average AUROC** for classification, and **≥ 3.2% reduction in MAE** for regression, outperforming all self-supervised foundation model baselines. Beyond accuracy, CLEF advances multifacet and robust single-lead ECG analysis, enabling next-generation remote health monitoring and wearable intelligence.

<div align="center">
  <img src="figures/model-overview.png" alt="CLEF Model Overview" width="70%"/>
</div>

## 🚀 News

- **[Nov 2025]** Paper released on arXiv
- **[Nov 2025]** Code and pretrained weights released

## ✨ Introduction

CLEF is an ECG foundation model trained with clinically-guided contrastive learning. This repository contains the implementation for our paper ["CLEF: Clinically-Guided Contrastive Learning for Electrocardiogram Foundation Models"](https://arxiv.org/abs/xxxxx).


**Key features:**
- Clinical-informed contrastive pretraining for better representation
- Pretrained models on 3 sizes: small, base, large
- Easy single-lead ECG representation extraction and downstream task evaluation

## 📚 Documentation

- [Environment Setup](doc/environment.md) — Docker and conda installation
- [Pretraining](doc/pretraining.md) — Run contrastive pretraining
- [Downstream Tasks](doc/downstream.md) — Evaluation on diagnostic tasks
- [Pretrained Weights](doc/get_pretrained_weight.md) — Download model checkpoints
- [Dataset Preparation](doc/dataset_preparation.md) — Prepare PTB-XL, ICENTIA, etc.
- [Example Notebooks](notebooks/) — Jupyter tutorials

## 🧭 Quick Start

- Launch the interactive quickstart notebook: `notebooks/clef_quickstart.ipynb`

## 📦 Available Models

| Model | Parameters | Download |
|-------|-----------|----------|
| CLEF-Small | 448K | [Zenodo](#) |
| CLEF-Base | 30.7M | [Zenodo](#) |
| CLEF-Large | 296M | [Zenodo](#) |


## 🙏 Acknowledgements

We gratefully acknowledge the contributions of the following projects, which were instrumental in the evaluation of CLEF:


* **Moment**: [moment-timeseries-foundation-model/moment](https://github.com/moment-timeseries-foundation-model/moment)
* **Moirai**: [SalesforceAIResearch/uni2ts](https://github.com/SalesforceAIResearch/uni2ts)
* **ECGFounder**: [PKUDigitalHealth/ECGFounder](https://github.com/PKUDigitalHealth/ECGFounder)
* **KED**: [control-spiderman/ECGFM-KED](https://github.com/control-spiderman/ECGFM-KED)
* **ST-MEM**: [bakqui/ST-MEM](https://github.com/bakqui/ST-MEM)
* **SimCLR**: [sthalles/SimCLR](https://github.com/sthalles/SimCLR)
* **BYOL**: [lucidrains/byol-pytorch](https://github.com/lucidrains/byol-pytorch)
* **MoCo**: [facebookresearch/moco](https://github.com/facebookresearch/moco)


## 📝 Citation

If you use CLEF in your research, please cite:

```bibtex
@article{clef2024,
  title={CLEF: Clinically-Guided Contrastive Learning for Electrocardiogram Foundation Models},
  author={Your Name},
  journal={arXiv preprint arXiv:xxxxx},
  year={2025}
}
```
