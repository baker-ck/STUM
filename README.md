## STUM-enhanced GNN Results

This repository contains trained baseline models (STGCN, AGCRN, GraphWaveNet, D2STGNN, STAEformer and STID) and enhanced STUM models for each trained baseline. Test results are provided for comparison. The implementation is based on the IEEE T-ITS 2025 paper “Cross Space and Time: A Spatio-Temporal Unitized Model for Traffic Flow Forecasting” (https://arxiv.org/pdf/2411.09251), with the following citation:

```bibtex
@article{ruan2025cross,
  title={Cross space and time: A spatio-temporal unitized model for traffic flow forecasting},
  author={Ruan, Weilin and Wang, Wenzhuo and Zhong, Siru and Chen, Wei and Liu, Li and Liang, Yuxuan},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2025},
  publisher={IEEE}
}
```

### Implementation notes
We use the publicly available STUM (https://github.com/RWLinno/STUM) codebase as our training framework. To ensure smooth execution, we applied minor engineering fixes (guarding optional wandb logging, correcting import paths, and fixing logging typos). No changes were made to model architectures, loss functions, or optimisation procedures. Each pretrained baseline model and enhanced model corresponds to a configuration file in baseline-configs/. and enhanced-configs/. Pretrained checkpoints are hosted externally due to size constraints.

### Repo structure
```
STUM/
├── README.md
├── baseline_configs/
│   ├── acgrn.yaml
│   ├── stgcn.yaml
│   ├── d2stgnn.yaml
│   ├── stae.yaml
│   ├── stum_vanilla.yaml
│   └── stid.yaml
├── baseline_models/
│   └── README.md   # external links to .pt files
├── enhanced_configs/
│   ├── stum_acgrn.yaml
│   ├── stum_stgcn.yaml
│   ├── stum_d2stgnn.yaml
│   ├── stum_stae.yaml
│   └── stum_stid.yaml
├── enhanced_models/
│   └── README.md   # external links to .pt files
├── stum_patches/
│   └── non_algorithmic_fixes.diff
└── stum_version.txt
```

### Environment
- Python:  3.8.8
- PyTorch: ---
- Device: MPS (Apple Silicon) 
- Conda environment exported in `environment.yml`

### Dataset
PEMS-BAY 07

### Results
| Trained baseline | Average MAE | Average RMSE | Average MAPE |
|-----------|-----------------------------|------------------------------|-------------------------------|
| STUM (vanilla)   |           |           |           |
| STGCN   |           |           |           |
| AGCRN  |   19.6861        |   32.7425       |  0.0828        |
| GWNet  |           |          |          |
| STID  |           |          |          |
| STAEformer  |           |          |          |
| D2STGNN  | 19.6793          |32.7329          |    0.0826      |

| Enhanced baseline | Average MAE | Average RMSE | Average MAPE |
|-----------|-----------------------------|------------------------------|-------------------------------|
| STUM_STGCN   |           |           |           |
| STUM_AGCRN  |           |          |          |
| STUM_GWNet  |           |          |          |
| STUM_STID  |           |          |          |
| STUM_STAEformer  |           |          |          |
| STUM_D2STGNN  |           |          |          |







