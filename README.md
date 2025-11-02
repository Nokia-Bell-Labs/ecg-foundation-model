<div align="center">
  <table border="0">
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

## 🚀 News

- **[Oct 2024]** Paper released on arXiv
- **[Oct 2024]** Code and pretrained weights released

## ✨ Introduction

CLEF is an ECG foundation model trained with clinically-guided contrastive learning. This repository contains the implementation for our paper ["CLEF: Clinically-Guided Contrastive Learning for Electrocardiogram Foundation Models"](https://arxiv.org/abs/xxxxx).


**Key features:**
- Metadata-guided contrastive pretraining (SimCLR, BYOL, MoCo)
- Support for 1-lead and 12-lead ECG configurations
- Pretrained models: small, base, large
- Easy downstream task evaluation

## 📚 Documentation

- [Environment Setup](doc/environment.md) — Docker and conda installation
- [Pretraining](doc/pretraining.md) — Run contrastive pretraining
- [Downstream Tasks](doc/downstream.md) — Evaluation on diagnostic tasks
- [Pretrained Weights](doc/get_pretrained_weight.md) — Download model checkpoints
- [Dataset Preparation](doc/dataset_preparation.md) — Prepare PTB-XL, ICENTIA, etc.
- [Example Notebooks](notebooks/) — Jupyter tutorials

## 📦 Available Models

| Model | Parameters | PTB-XL (Macro F1) | Download |
|-------|-----------|-------------------|----------|
| CLEF-Small | 448K | 0.XX | [Zenodo](#) |
| CLEF-Base | 30.7M | 0.XX | [Zenodo](#) |
| CLEF-Large | 296M | 0.XX | [Zenodo](#) |


## 📝 Citation

If you use CLEF in your research, please cite:

```bibtex
@article{clef2024,
  title={CLEF: Clinically-Guided Contrastive Learning for Electrocardiogram Foundation Models},
  author={Your Name},
  journal={arXiv preprint arXiv:xxxxx},
  year={2024}
}
```

## 📄 License