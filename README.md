
# M<sup>3</sup>Net: Meta-Reinforcement Learning-Based Open-Set Domain Generalization for HSI Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-≥1.7.0-red.svg)](https://pytorch.org)

Official PyTorch implementation of the paper **"M<sup>3</sup>Net: Meta-Reinforcement Learning-Based Open-Set Domain Generalization of Hyperspectral Image Classification Model"**

---

## 🚀 Quick Start

### 📦 Requirements
```bash
# Core dependencies
pip install torch>=1.7.0 numpy scipy matplotlib tqdm thop scikit-learn
```

### 🗂 Dataset Preparation
**Download datasets from:** [Cloud Storage](https://www.jianguoyun.com/p/DSs6tk4Q4pXJDBiagvMFIAA)

**Directory structure:**
```
dataset/
├── Houston/
│   ├── Houston13.mat
│   ├── Houston13_7gt.mat
│   ├── Houston18.mat
│   └── Houston18_7gt.mat
├── Pavia/
│   ├── paviaC.mat
│   ├── paviaC_7gt.mat
│   ├── paviaU.mat
│   └── paviaU_7gt.mat
└── BOT/
    ├── BOT5.mat
    ├── BOT5_gt.mat
    ├── BOT6.mat
    └── BOT6_gt.mat
```

### 🏗 Project Structure
```
M3Net/
├── code/
│   ├── train.py           # Main training pipeline
│   ├── model.py           # Neural architecture
│   ├── data_manager.py    # Data preprocessing
│   └── utils.py           # Helper functions
└── dataset/               # Preprocessed datasets
```

---

## 📖 Citation
If you use this work in your research, please cite:
```bibtex
@article{m3net2025,
  title={M<sup>3</sup>Net: Meta-Reinforcement Learning-Based Open-Set Domain Generalization for HSI Classification},
  author={},
  journal={},
  year={2025},
  doi={}
}
```

---

## 📜 License
This project is open source under [MIT License](LICENSE).
