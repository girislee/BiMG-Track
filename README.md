# BiMG-Track

Official implementation of the paper:

BiMG-Track: Bidirectional Temporal Modeling and Adaptive Multi-Granularity Interactor for Robust RGB-T Tracking

Wenjuan Li, Ke Wang, Jun Tang, Dong Liang, Lilong Duan


📌 Introduction

RGB-T tracking aims to robustly localize a target by fusing visible-light (RGB) and thermal infrared (TIR) information. Existing methods often struggle with:

- insufficient modeling of dynamic cross-modal spatio-temporal dependencies;
- unreliable target discrimination when RGB and TIR cues are inconsistent;
- temporal misalignment and visual ambiguity under illumination changes, occlusion, and thermal crossover.

To address these challenges, we propose BiMG-Track, a robust RGB-T tracking framework based on:

1. Bidirectional Temporal Aggregation Block (BTA Block)  
   Built upon a Bidirectional Parallel LSTM (BPLSTM) with cross-modal reliability fusion, BTA captures long-range temporal dependencies and mitigates global temporal misalignment between RGB and TIR modalities.

2. Adaptive Multi-Granularity Interactor (AMGI)
   Equipped with a Dynamic Sparse Mask (DSM) and a Dual-Path Attention (DPA) mechanism, AMGI suppresses redundant features while balancing RGB-dominant local texture/edge cues and TIR-dominant global semantic representations.

Extensive experiments on LasHeR, RGBT234, RGBT210, and VTUAV demonstrate that BiMG-Track achieves state-of-the-art performance, especially under modality inconsistency and complex background interference.

---

✨ Highlights

- Unified end-to-end RGB-T tracking framework.
- Bidirectional parallel temporal modeling for cross-modal spatio-temporal aggregation.
- Cross-modal reliability fusion in the temporal latent space.
- Dynamic sparse masking for redundancy suppression.
- Modality-specific dual-path attention: RGB → IPA, TIR → CPA.
- Strong performance on four public RGB-T tracking benchmarks.

---

🧠 Method Overview

BiMG-Track adopts a dual-stream collaborative architecture for RGB and TIR modalities.

- RGB and TIR frames are split into `16 × 16` patches and projected into token embeddings.
- Tokens are fed into `N` BTA Blocks.
- Each BTA Block contains:
  - a BPLSTM for bidirectional temporal modeling and cross-modal reliability fusion;
  - a ViT spatial encoder for intra-frame spatial modeling;
  - an AMGI for spatial-domain multi-granularity interaction.
- The fused features are finally passed to a prediction head for bounding box regression and classification.

