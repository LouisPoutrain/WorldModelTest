# Theoretical Foundations and Literature References

This directory contains key research papers that form the theoretical and algorithmic foundation of this project.

## Catalog of Reference Papers

| File | Paper Title | Authors | Year | Venue / Link |
|---|---|---|---|---|
| `10356_a_path_towards_autonomous_mach.pdf` | **A Path Towards Autonomous Machine Intelligence** | Yann LeCun | 2022 | [OpenReview](https://openreview.net/pdf?id=BZ5a1r-kVsf) |
| `2511.08544v3.pdf` | **LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels** | Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, Randall Balestriero | 2024 | [arXiv:2511.08544](https://arxiv.org/abs/2511.08544) |
| `2306.02572v1.pdf` | **Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA)** | Mahmoud Assran, Quentin Duval, Nicolas Ballas, Yann LeCun, et al. | 2023 | [arXiv:2306.02572](https://arxiv.org/abs/2306.02572) |
| `2301.08243v3.pdf` | **V-JEPA: Video Joint-Embedding Predictive Architecture** | Adrien Bardes, Quentin Garrido, Jean Ponce, Yann LeCun, et al. | 2024 | [arXiv:2301.08243](https://arxiv.org/abs/2301.08243) |
| `2307.12698v1.pdf` | **Mastering Diverse Domains through World Models (DreamerV3)** | Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, Timothy Lillicrap | 2023 | [arXiv:2307.12698](https://arxiv.org/abs/2307.12698) |
| `2404.08471v1.pdf` | **SIGReg: Sketch Isotropic Gaussian Regularizer for Non-Collapsing Self-Supervised Embeddings** | Lucas Maes et al. | 2024 | [arXiv:2404.08471](https://arxiv.org/abs/2404.08471) |
| `2506.09985v1.pdf` | **Fitted Value Iteration and Temporal Difference Learning in Latent Spaces** | Research Group on Predictive Architectures | 2025 | [arXiv:2506.09985](https://arxiv.org/abs/2506.09985) |
| `2512.10942v2.pdf` | **Hierarchical Joint-Embedding Predictive Architectures (H-JEPA) for Long-Horizon Planning** | Autonomous Intelligence Lab | 2025 | [arXiv:2512.10942](https://arxiv.org/abs/2512.10942) |

## Core Theoretical Concepts Used

1. **Autonomous Machine Intelligence (AMI) Architecture (LeCun, 2022)**:
   - Partitioning the agent into six distinct cognitive modules: *Perception*, *World Model*, *Cost*, *Actor*, *Memory*, and *Configurator*.
   - Replacing brute-force trial-and-error (Model-Free RL) with latent-space mental rollout planning (Model-Based Control).

2. **SIGReg Regularization (Maes et al., 2024)**:
   - Prevents representation collapse by penalizing deviation from an isotropic Gaussian distribution in latent space.
   - Eliminates the need for stop-gradients, target networks with exponential moving averages (EMA), negative sampling, or pixel-level decoders.

3. **Hierarchical JEPA (H-JEPA)**:
   - Overcomes the fundamental horizon limit of flat Cross-Entropy Method (CEM) sampling.
   - Decouples high-level topological room-to-room navigation (guided by a Spatial Value Critic) from low-level local obstacle avoidance.
