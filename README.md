# M3Net: Meta-Reinforcement Learning-Based Open-Set Domain Generalization of Hyperspectral Image Classification Model
Official PyTorch implementation of the paper《M3Net: Meta-Reinforcement Learning-Based Open-Set Domain Generalization of Hyperspectral Image Classification Model》

## 🚀 fast begining

### Requirements
torch >= 1.7.0
numpy
scipy
matplotlib
tqdm
thop
scikit-learn

### Dataset Preparation
**Organize datasets in the following directory structure under `dataset/`:**

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

### Project Structure
```
M3Net/
├── code/
│   ├── train.py           # Main training script
│   ├── model.py           # Model architecture definition
│   ├── data_manager.py    # Data loading and preprocessing
│   └── utils.py           # Helper functions and utilities
└── dataset/               # Dataset storage directory
```



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
