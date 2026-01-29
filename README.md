<div align="center">

# UAD-Flow: Unified Autonomous Driving with Flow-based World Model

[![Paper](https://img.shields.io/badge/Paper-OpenReview-blue)](https://openreview.net/pdf?id=m4JpoJRgAr)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-red)](https://openreview.net/pdf?id=m4JpoJRgAr)

</div>

## Introduction

This repository contains the official implementation of **UAD-Flow**, a unified framework that integrates flow-based world models into various autonomous driving perception and planning systems. Our method enhances the temporal understanding and prediction capabilities of existing models through a novel flow-based attention mechanism.

### Key Features

- **Flow-based World Model**: Leverages optical flow information to enhance temporal feature aggregation across multi-view cameras
- **Unified Framework**: Applicable to various autonomous driving tasks including 3D detection, occupancy prediction, motion planning, and VLM-based driving
- **Multi-scale Flow Patches**: Hierarchical flow feature extraction with configurable patch sizes (`flow_patches`) and feature levels (`flow_ids`)
- **Seamless Integration**: Drop-in enhancement for existing autonomous driving models

### Core Technical Contribution

Our flow method introduces a cross-view feature flow mechanism that:
1. Aggregates features from adjacent camera views using learnable flow MLPs
2. Applies multi-scale unfold-fold operations to capture spatial-temporal correlations
3. Enhances the original feature maps with flow-aware representations

Key parameters:
- `flow_patches`: Patch sizes for multi-scale flow extraction (e.g., `[8, 4, 2, 1]`)
- `flow_ids`: Feature level indices for flow processing (e.g., `[0, 1, 2, 3]`)

## Project Structure

This repository contains four sub-projects, each applying the flow method to a different base model:

| Project | Base Model | Task | Description |
|---------|------------|------|-------------|
| [Senna-Flow](./Senna-Flow/) | [Senna](https://github.com/hustvl/Senna) | VLM-based Driving | Flow-enhanced vision-language model for end-to-end autonomous driving |
| [SparseBEV-Flow](./SparseBEV-Flow/) | [SparseBEV](https://github.com/MCG-NJU/SparseBEV) | 3D Object Detection | Flow-enhanced sparse BEV detection |
| [SparseDrive-Flow](./SparseDrive-Flow/) | [SparseDrive](https://github.com/swc-17/SparseDrive) | End-to-End Driving | Flow-enhanced sparse scene representation for planning |
| [SparseOcc-Flow](./SparseOcc-Flow/) | [SparseOcc](https://github.com/MCG-NJU/SparseOcc) | Occupancy Prediction | Flow-enhanced sparse occupancy prediction |

## Getting Started

### Prerequisites

Each sub-project has its own environment requirements. Please refer to the individual README files for detailed setup instructions.

### Quick Start

1. Clone this repository:
```bash
git clone https://github.com/your-repo/UAD-Flow.git
cd UAD-Flow
```

2. Navigate to the desired sub-project:
```bash
cd SparseBEV-Flow  # or Senna-Flow, SparseDrive-Flow, SparseOcc-Flow
```

3. Follow the sub-project's README for environment setup and training/evaluation.

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@inproceedings{uad-flow,
  title={UAD-Flow: Unified Autonomous Driving with Flow-based World Model},
  author={},
  booktitle={},
  year={2025},
  url={https://openreview.net/pdf?id=m4JpoJRgAr}
}
```

## Acknowledgements

This work builds upon several excellent open-source projects:

- [Senna](https://github.com/hustvl/Senna) - VLM-based autonomous driving
- [SparseBEV](https://github.com/MCG-NJU/SparseBEV) - Sparse 3D object detection
- [SparseDrive](https://github.com/swc-17/SparseDrive) - End-to-end autonomous driving
- [SparseOcc](https://github.com/MCG-NJU/SparseOcc) - Sparse occupancy prediction
- [LLaVA](https://github.com/haotian-liu/LLaVA) - Large language and vision assistant
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) - 3D detection toolbox

## License

This project is released under the Apache 2.0 license. Please see the LICENSE files in each sub-project for more details.
