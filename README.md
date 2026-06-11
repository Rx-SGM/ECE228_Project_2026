# Device-Free WiFi-Based Biometric Authentication with Neural Networks

> Passive, device-free gait authentication using WiFi Channel State Information (CSI) Doppler spectrograms and a metric-learning CNN — no wearables, no network modifications required.

**ECE 228: Machine Learning for Physical Applications | Spring 2026 | UC San Diego**
**Authors:** Soham Guha Mazumder · Afreen Haider

---

## Table of Contents
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Running the Code](#running-the-code)
- [Results](#results)
- [Dependencies](#dependencies)

---

## Overview
In this project, we aim to perform CSI-based user recognition with the use of a Siamese neural network trained on Doppler-time spectrograms extracted from ambient 802.11n WiFi Channel State Information (CSI). Specifically, we frame the problem as an open-set biometric authentication, i.e., given a set of registered users whose gait signatures are enrolled during training, the system must decide at
inference time whether a new CSI observation belongs to a registered user or to an unrecognized intruder without assuming the query identity is among the known classes of individuals.

This project builds an **open-set biometric authentication system** that identifies and verifies people purely from how they walk through a WiFi link. When a person walks between a WiFi transmitter and receiver, their body perturbs the multipath channel in a gait-specific pattern captured in the Channel State Information (CSI). We process this into a Doppler-time spectrogram and train a CNN embedding network using **Supervised Contrastive (SupCon) Loss** to learn a compact 128-dimensional representation of each person's gait.

At inference time, an unknown query spectrogram is either **accepted** as a registered user or **rejected** as an intruder based on its nearest-neighbour distance to a pre-enrolled gallery — with no prior knowledge of the intruder's identity.

---

## Repository Structure

```
.
├── preprocessing_CSI_data.ipynb  
├── Siamese_network_with_results.ipynb   
├── csi_dataset
├── Manuscript_data
└── README.md

```

### Notebook Descriptions

`preprocessing_CSI_data.ipynb` : Reads raw WiStride CSV files; performs AP selection, complex subcarrier extraction, walk segmentation, uniform resampling, static multipath removal, and STFT spectrogram generation. Outputs a single `spectrograms.npz` file used by the model notebook. 
`Siamese_network_with_results.ipynb` : Loads spectrograms, trains GaitNet with SupCon loss, builds the expanded gallery, runs LOCO threshold calibration, and evaluates open-set authentication performance. All plots and per-user breakdowns are included inline. 
`Siamese_network_with_results.ipynb` : Loads spectrograms, trains GaitNet with SupCon loss, builds the expanded gallery, runs LOCO threshold calibration, and evaluates open-set authentication performance. All plots and per-user breakdowns are included inline. 
`Siamese_network_with_results.ipynb` : Loads spectrograms, trains GaitNet with SupCon loss, builds the expanded gallery, runs LOCO threshold calibration, and evaluates open-set authentication performance. All plots and per-user breakdowns are included inline. 

---

## Dataset

We use the **[WiStride](https://ieee-dataport.org/documents/wistride-csi-based-gait-dataset-indoor-walking-subjects)** dataset 

---

## Running the Code

Both notebooks are designed to run on **Google Colab** with GPU acceleration. No local installation is required beyond uploading the files to Google Drive.

### Step 1 — Preprocessing

Execute `preprocessing_CSI_data.ipynb`. This reads the raw WiStride CSVs and writes `spectrograms1.npz` to the system.

### Step 2 — Training and Evaluation

Execute `Siamese_network_with_results.ipynb` in Colab (**GPU runtime recommended**)

| Cell | Action |
|------|--------|
| Setup | Configure hyperparameters (`TEMPERATURE`, `EPOCHS`, etc.) |
| Load Dataset | Load `spectrograms1.npz` |
| Data Split | 70 / 15 / 15 chronological split |
| Training | Train GaitNet for 400 epochs (~15–20 min on T4 GPU) |
| Gallery | Build 653-point expanded gallery |
| Calibration | Run LOCO threshold calibration |
| Evaluation | Print auth/intruder results at EER and 10% FRR |

The trained model is saved to `models/gait_net_v3.pth`.

---

## Results

| Metric | @ EER (δ = 0.192) | @ 10% FRR (δ = 0.409) |
|--------|:-----------------:|:---------------------:|
| Auth acceptance rate | 52.8% | 86.4% |
| Intruder rejection rate | **46.7%** | 10.2% |
| Identification accuracy | 8.8% | 13.6% |
| Training embedding separation (Δ) | **1.339** | — |
| Test separation (intruder − auth) | **+0.004** | — |

The resulting plots can be seen in `plots`


## Dependencies

```
torch >= 2.0
numpy
matplotlib
scikit-learn
scipy

```

---


