# M3Net: Open-Set Domain Generalization for HSI Classification
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

论文《Meta-Reinforcement Learning-Based Open-Set Domain Generalization of Hyperspectral Image Classification Model》的PyTorch实现。

## 🚀 fast begining

### Requirements
torch >= 1.7.0
numpy
scipy
matplotlib
tqdm
thop
scikit-learn

### 数据准备
**Datasets should be placed in the `dataset/` directory with the following structure:**

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

### Project Structure
M3Net/
├── code/
│   ├── train.py           # Main training script
│   ├── model.py           # Model definition
│   ├── data_manager.py    # Data loading and preprocessing
│   └── utils.py           # Utility functions
└── dataset/               # Dataset directory


### Citation
If you use this code, please cite the following paper:
```bibtex
@article{m3net2025,
  title={Meta-Reinforcement Learning-Based Open-Set Domain Generalization for HSI Classification},
  author={Author1, Author2},
  year={2025}
}
```

## LICENSE
[MIT License](LICENSE)
