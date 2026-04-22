# FastDVDnet: Training & Evaluation Guide

This repository contains scripts for training and evaluating **FastDVDnet** using self-supervised frameworks like **R2R** (Recorrupted-to-Recorrupted) and **SURE/PURE**.

---

## 1. trains.sh (Batch Hyperparameter Search)
Used to launch multiple training sessions (Grid Search) sequentially.

### Example: Synthetic FMDD Training (with R2R-Poisson)
```bash
./trains.sh "1e-5 1e-6" "0.15 0.3" 100 r2r_p 20 5 false fmdd synthetic 1.0
```
*   **"1e-5 1e-6"**: Learning Rates to test.
*   **"0.15 0.3"**: Alpha values (recorruption strength) to test.
*   **100**: Maximum number of epochs.
*   **r2r_p**: Loss type (Poisson R2R).
*   **20**: Patience for Early Stopping.
*   **fmdd synthetic**: FMDD dataset in synthetic noise generation mode.
*   **1.0**: Poisson gain ($\gamma$).

---

## 2. train.py (Core Training Script)
Direct call for a specific experiment configuration.

### Example: Loreal Real Data (with PURE)
```bash
python train.py --epochs 50 --lr 1e-5 --loss pure --tau1 0.0005 --dataset_type loreal --data_scale 255
```
*   **--loss pure**: Risk estimator for real Poisson noise.
*   **--tau1**: Divergence value for the SURE/PURE regularizer.

---

## 3. test4.py (Single Sequence Evaluation)
Evaluates a specific sequence with optional Self-Ensemble support.

### Example: Synthetic Test (Single Image) with Self-Ensemble
```bash
python test4.py --input data/FMDD/WideField/gt/4/avg50.png --synthetic_test --gamma 1.0 --n_samples 8 --network results/train_.../ckpts/best_model.pth --output results/test_denoised_%03d.tif
```
*   **--synthetic_test**: Injects Poisson noise into the clean input on-the-fly.
*   **--n_samples 8**: Performs 8 forward passes (Self-Ensemble) and averages them for higher quality.
*   **--input**: Can be a static image (like `avg50.png`) or a sequence.
*   **Outputs**: Automatically saves `input_noisy_ns8_000.tif` along with the denoised result for easy comparison.

---

## 4. test_experiments.sh (Batch Experiment Evaluation)
Automatically evaluates **all** experiments within a results folder against a test sequence.

### Example: Evaluate all Grid Search models on FMDD
```bash
./test_experiments.sh results/train_26-04-21... data/FMDD/gt/12/avg50.png "None" 255 0 0 "--synthetic_test --gamma 1.0 --n_samples 8"
```
*   **Directory**: Path containing the `lr_..._alpha_...` subdirectories.
*   **"None"**: Indicates no pre-processing file is needed (automatically handled).
*   **"--..."**: Extra arguments are passed directly to `test4.py`.
