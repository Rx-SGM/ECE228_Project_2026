# WiFi Gait Authentication via Supervised Contrastive Learning

> Passive, device-free gait authentication using WiFi Channel State Information (CSI) Doppler spectrograms and a metric-learning CNN — no wearables, no network modifications required.

**ECE 228 — Machine Learning for Physical Applications | Spring 2026 | UC San Diego**
**Authors:** Soham Guha Mazumder · Afreen Haider

---

## Table of Contents
- [Overview](#overview)
- [How It Works](#how-it-works)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Running the Code](#running-the-code)
- [Results](#results)
- [Dependencies](#dependencies)

---

## Overview

This project builds an **open-set biometric authentication system** that identifies and verifies people purely from how they walk through a WiFi link. When a person walks between a WiFi transmitter and receiver, their body perturbs the multipath channel in a gait-specific pattern captured in the Channel State Information (CSI). We process this into a Doppler-time spectrogram and train a CNN embedding network using **Supervised Contrastive (SupCon) Loss** to learn a compact 128-dimensional representation of each person's gait.

At inference time, an unknown query spectrogram is either **accepted** as a registered user or **rejected** as an intruder based on its nearest-neighbour distance to a pre-enrolled gallery — with no prior knowledge of the intruder's identity.

---

## How It Works

```
Raw CSI packets (passive WiFi sniffing)
        │
        ▼
 AP Selection (highest effective sampling rate ≥ 48 Hz)
        │
        ▼
 Complex subcarrier extraction (52 valid subcarriers)
        │
        ▼
 Walk segmentation → Uniform resampling → Static multipath removal
        │
        ▼
 STFT Doppler-time spectrogram → resize 64×64 → z-score normalise
        │
        ▼
 GaitNet CNN  (4-block encoder + projection head → 128-dim ℓ₂ embedding)
        │
    Trained with SupCon Loss  (300 positive pairs/batch vs ~8 for triplet)
        │
        ▼
 Expanded gallery enrollment  (653 reference points, ~33/user)
        │
        ▼
 LOCO threshold calibration  (EER threshold δ* = 0.192)
        │
        ▼
 1-NN query → ACCEPT / REJECT
```

---

## Repository Structure

```
.
├── preprocessing_CSI_data.ipynb   # Stage 1: CSI preprocessing to get spectrograms
├── Siamese_network_with_results.ipynb      # Stage 2: GaitNet plus SupCon --> training, evaluation, results
└── README.md
```

### Notebook Descriptions

| Notebook | Purpose |
|----------|---------|
| `preprocessing_CSI_data.ipynb` | Reads raw WiStride CSV files; performs AP selection, complex subcarrier extraction, walk segmentation, uniform resampling, static multipath removal, and STFT spectrogram generation. Outputs a single `spectrograms.npz` file used by the model notebook. |
| `Siamese_network_with_results.ipynb` | Loads spectrograms, trains GaitNet with SupCon loss, builds the expanded gallery, runs LOCO threshold calibration, and evaluates open-set authentication performance. All plots and per-user breakdowns are included inline. |

---

## Dataset

We use the **[WiStride](https://ieee-dataport.org/documents/wistride-csi-based-gait-dataset-indoor-walking-subjects)** dataset — a publicly available CSI-based gait dataset.

| Property | Value |
|----------|-------|
| Participants | 25 (20 registered + 5 intruders) |
| Walks per participant | ~40 |
| Total spectrograms | 975 |
| Environment | Static indoor corridor |
| Hardware | Passive 802.11n WiFi sniffer, 2.4 GHz |
| Spectrogram size | 64 × 64 pixels |

Make sure to change the load path and the save path according to your use.

Then run the preprocessing notebook to generate `spectrograms1.npz`.

---

## Running the Code

Both notebooks are designed to run on **Google Colab** with GPU acceleration. No local installation is required beyond uploading the files to Google Drive.
The Colab notebook was mounted by Google Drive, and all loading of the datasets, and the storage of the saved models was done on Google Drive.

### Step 1 — Preprocessing


Open `preprocessing_CSI_data.ipynb` in Colab and run all cells.  
This reads the raw WiStride CSVs and writes `spectrograms1.npz` to your Drive.



### Step 2 — Training and Evaluation

Open `Siamese_network_with_results.ipynb` in Colab (**GPU runtime recommended**) and run all cells in order.

| Cell | Action |
|------|--------|
| Setup | Configure hyperparameters (`TEMPERATURE`, `EPOCHS`, etc.) |
| Load Dataset | Load `spectrograms1.npz` |
| Data Split | 70 / 15 / 15 chronological split |
| Training | Train GaitNet for 400 epochs (~15–20 min on T4 GPU) |
| Gallery | Build 653-point expanded gallery |
| Calibration | Run LOCO threshold calibration |
| Evaluation | Print auth/intruder results at EER and 10% FRR |

The trained model is saved to `models/gait_net_v3.pth` on Drive.

---

## Results

| Metric | @ EER (δ = 0.192) | @ 10% FRR (δ = 0.409) |
|--------|:-----------------:|:---------------------:|
| Auth acceptance rate | 52.8% | 86.4% |
| Intruder rejection rate | **46.7%** | 10.2% |
| Identification accuracy | 8.8% | 13.6% |
| Training embedding separation (Δ) | **1.339** | — |
| Test separation (intruder − auth) | **+0.004** | — |

Key findings:
- SupCon loss achieves a training embedding separation of **Δ = 1.339**, nearly 10× higher than the triplet-loss baseline (Δ ≈ 0.14)
- The expanded gallery (653 pts) raises intruder rejection by **+11 percentage points** over the 112-point baseline and is the only configuration that maintains **positive test separation**
- LOCO calibration produces a threshold at the correct distance scale (0.192) vs. pairwise calibration (> 1.0)


## Dependencies

All standard Colab libraries. For local use:

```
torch >= 2.0
numpy
matplotlib
scikit-learn
scipy

```

---


