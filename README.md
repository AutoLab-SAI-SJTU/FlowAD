<div align="center">
<img src="assets/logo.png" width="300">
<h3>FlowAD: Ego-Scene Interactive Modeling for Autonomous Driving</h3>

Mingzhe Guo<sup>1,2</sup>, Yixiang Yang<sup>1</sup>, Chuanrong Han<sup>1</sup>, Rufeng Zhang<sup>2</sup>, Shirui Li<sup>2</sup>, Ji Wan<sup>2</sup>, Zhipeng Zhang<sup>1 ✉</sup>

<sup>1</sup> AutoLab, School of Artificial Intelligence, Shanghai Jiao Tong University, <sup>2</sup> Baidu Inc.

<sup>✉</sup> corresponding author: zhipeng.zhang.cv@outlook.com

Accepted to ICLR 2026!


[![Paper](https://img.shields.io/badge/Paper-OpenReview-blue)](https://openreview.net/pdf?id=m4JpoJRgAr)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-red)](https://openreview.net/pdf?id=m4JpoJRgAr)

</div>

## Table of Contents

- [Introduction](#introduction)
- [Performance Highlights](#performance-highlights)
  - [nuScenes Open-Loop Evaluation](#nuscenes-open-loop-evaluation)
  - [Qualitative Results - Perception](#qualitative-results---perception)
  - [Bench2Drive Closed-Loop Evaluation](#bench2drive-closed-loop-evaluation)
- [Getting Started](#getting-started)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)


## Introduction

This repository contains the official implementation of **FlowAD**, a novel ego-scene interactive modeling framework for autonomous driving. Unlike traditional approaches that treat each timestamp in isolation, FlowAD explicitly models the **feedback of ego-vehicle motion** to future observations, fundamentally improving the understanding of the driving process and enhancing planning capabilities.

![The architecture of our FlowAD](assets/flowad_pipeline.png) 
*The architecture of our FlowAD structured around three core components: 1) Ego-guided scene partition. 2) Spatial and temporal flow prediction. 3) Task-aware enhancement.*

Inspired by human perception, FlowAD represents ego-scene interaction as **scene flow relative to the ego-vehicle**, capturing relative motion as learnable scene flow within the latent feature space. This enables modeling ego-motion feedback using existing log-replay datasets without requiring complex scenario simulations.

**Key Achievements:**
- **19% collision rate reduction** over SparseDrive on nuScenes
- **60% FCP (our proposed metric) improvement** (1.39 frames) on nuScenes validation set  
- **51.77 driving score** on Bench2Drive closed-loop evaluation
- Demonstrated generality across **perception, end-to-end planning, and VLM analysis**


<!-- ## Project Structure

This repository contains four sub-projects, each applying the flow method to a different base model:

| Project | Base Model | Task | Description |
|---------|------------|------|-------------|
| [Senna-Flow](./Senna-Flow/) | [Senna](https://github.com/hustvl/Senna) | VLM-based Driving | Flow-enhanced vision-language model for end-to-end autonomous driving |
| [SparseBEV-Flow](./SparseBEV-Flow/) | [SparseBEV](https://github.com/MCG-NJU/SparseBEV) | 3D Object Detection | Flow-enhanced sparse BEV detection |
| [SparseDrive-Flow](./SparseDrive-Flow/) | [SparseDrive](https://github.com/swc-17/SparseDrive) | End-to-End Driving | Flow-enhanced sparse scene representation for planning |
| [SparseOcc-Flow](./SparseOcc-Flow/) | [SparseOcc](https://github.com/MCG-NJU/SparseOcc) | Occupancy Prediction | Flow-enhanced sparse occupancy prediction | -->



## Performance Highlights

### nuScenes Open-Loop Evaluation

Our method achieves significant improvements across multiple tasks:

| Method | Backbone | Detection |  | Tracking |  | Motion Prediction |  | Planning |  | FCP↓ |
|--------|----------|-----------|--|----------|--|-------------------|--|----------|--|------|
|        |          | mAP↑ | NDS↑ | AMOTA↑ | AMOTP↓ | minADE↓ | minFDE↓ | Avg.L2 (m)↓ | Avg.Col↓ |  |
| UniAD  | ResNet101 | 0.380 | 0.498 | 0.359 | 1.320 | 0.71 | 1.02 | 0.69 | 0.12 | 2.96 |
| SparseDrive | ResNet101 | 0.496 | 0.588 | 0.501 | 1.085 | 0.60 | 0.96 | 0.58 | 0.06 | 2.30 |
| **FlowAD (Ours)** | **ResNet101** | **0.523** | **0.605** | **0.518** | **1.040** | **0.56** | **0.93** | **0.52** | **0.05** | **0.91** |

*FCP (Frames before Correct Planning): Lower is better. FlowAD achieves 1.39 frames improvement (48% reduction) over SparseDrive and 2.03 frames improvement (60% reduction) over baseline methods.*

### Qualitative Results - Perception

<!-- TODO: Add perception visualization here -->
![Perception Results](assets/vis-detection.png) 
*FlowAD demonstrates superior detection of occluded objects, small targets, and dense scenes through learned scene flow dynamics.*

### Bench2Drive Closed-Loop Evaluation

![Bench2Drive Results](assets/vis-b2d.png)
*FlowAD achieves **51.77 driving score**, demonstrating robust closed-loop performance.*


## Getting Started


### Quick Start

1. Clone this repository:
```bash
git clone https://github.com/your-repo/FlowAD.git
cd FlowAD
```

2. Navigate to the desired sub-project:
```bash
cd SparseDrive-Flow 
```

3. Follow the sub-project's [README](SparseDrive-Flow/README.md) for environment setup and training/evaluation.

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@inproceedings{FlowAD2026,
  title={FlowAD: Ego-Scene Interactive Modeling for Autonomous Driving},
  author={Anonymous},
  booktitle={Under review as a conference paper at ICLR 2026},
  year={2026},
  url={https://openreview.net/pdf?id=m4JpoJRgAr}
}
```

## Acknowledgements

This work builds upon several excellent open-source projects:

<!-- - [Senna](https://github.com/hustvl/Senna) - VLM-based autonomous driving
- [SparseBEV](https://github.com/MCG-NJU/SparseBEV) - Sparse 3D object detection -->
- [SparseDrive](https://github.com/swc-17/SparseDrive) - End-to-end autonomous driving
<!-- - [SparseOcc](https://github.com/MCG-NJU/SparseOcc) - Sparse occupancy prediction -->

## License

This project is released under the Apache 2.0 license. Please see the LICENSE files in each sub-project for more details.
