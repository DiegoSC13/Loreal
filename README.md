# FastDVDnet: Training & Evaluation Guide

This repository contains scripts for training and evaluating **FastDVDnet** using self-supervised frameworks like **R2R** (Recorrupted-to-Recorrupted) and **SURE/PURE**.

---

## 1. main.sh (Batch Hyperparameter Search)
Used to launch multiple training sessions (Grid Search) sequentially.

### Example: Loreal Real Data Training
```bash
./main.sh "1e-4 1e-5" "0.005 0.001" 50 pure 10 25 false loreal raw 1.0
```
*   **"1e-4 1e-5"**: Learning Rates to test.
*   **"0.005 0.001"**: Hyperparameter to test (Tau1 for PURE/SURE, or alpha for R2R).
*   **50**: Maximum number of epochs.
*   **pure**: Loss type.
*   **10**: Patience for Early Stopping.
*   **25**: Number of test samples for evaluation.
*   **false**: Geometric TTA disabled.
*   **loreal raw 1.0**: Dataset type (`loreal`), mode (`raw`), and Gamma value (`1.0`).

### Example: Synthetic FMDD Training (with R2R-Poisson)
```bash
./main.sh "1e-5 1e-6" "0.15 0.3" 100 r2r_p 20 5 false fmdd synthetic 1.0
```
*   **fmdd synthetic**: FMDD dataset in synthetic noise generation mode.

---

## 2. train.py (Core Training Script)
Direct call for a specific experiment configuration.

### Example: Loreal Real Data (with PURE)
```bash
python train.py --sequence_directory ../data/Loreal --dataset_type loreal --fmdd_mode raw --epochs 50 --lr 1e-5 --loss pure --tau1 0.0005 --data_scale 255 --output_path ./results/my_exp
```

### Example: Synthetic FMDD Data
```bash
python train.py --sequence_directory ../data/FMDD --dataset_type fmdd --fmdd_mode synthetic --epochs 100 --lr 1e-6 --loss r2r_p --alpha 0.15 --gamma 1.0 --data_scale 255 --output_path ./results/my_exp
```

---

## 3. test4.py (Single Sequence Evaluation)
Evaluates a specific sequence with optional Self-Ensemble support.

### Example: FMDD Synthetic Test (Single Image)
```bash
python test4.py --input ../data/FMDD/WideField_BPAE_G/gt/12/avg50.png --first 0 --last 0 --synthetic_test --gamma 1.0 --n_samples 8 --network results/train_.../ckpts/best_model.pth --output results/test_denoised_%03d.tif --data_scale 255
```
*   **--synthetic_test**: Injects Poisson noise into the clean input on-the-fly.
*   **--n_samples 8**: Performs 8 forward passes (Self-Ensemble) and averages them for higher quality.

### Example: Loreal Real Data Test (Sequence)
```bash
python test4.py --input ../data/Loreal/sequence_1/image_%03d.tif --first 0 --last 29 --network results/train_.../ckpts/best_model.pth --output results/test_denoised_%03d.tif --data_scale 255
```

---

## 4. test_experiments.sh (Batch Experiment Evaluation)
Automatically evaluates **all** experiments within a results folder against a test sequence. Fallbacks are dynamic depending on the base_dir name.

### Example: Evaluate FMDD Experiments (Synthetic Default)
```bash
# Uses smart defaults for FMDD (synthetic_test, gamma 1.0)
./test_experiments.sh results/train_26-04-26_15-22-10_r2r_p_fmdd "../data/FMDD/WideField_BPAE_G/gt/12/avg50.png" "none" 255
```

### Example: Evaluate Loreal Experiments
```bash
# Uses smart defaults for Loreal
./test_experiments.sh results/train_Loreal_Exp
```

---

## 5. compute_metrics.py (Quantitative Comparison)
Calculates PSNR and SSIM across all sub-experiments in a directory and exports a summary CSV.

### Example: Compute metrics for FMDD (Single GT image)
```bash
python compute_metrics.py --results_dir results/train_26-04-26_..._fmdd --gt ../data/FMDD/WideField_BPAE_G/gt/12/avg50.png --first 0 --last 0 --save_diff
```
*   **--save_diff**: Generates normalized difference maps (`diff_000.tif`) in each experiment folder.

### Example: Compute metrics for Loreal (Sequence of GT images)
```bash
python compute_metrics.py --results_dir results/train_Loreal --gt ../data/Loreal/sequence_1/gt_image_%03d.tif --first 0 --last 29
```

---

## 6. plot_experiments.py (Visualizing Training Curves)
Generates combined plot graphs for loss, PSNR, or SSIM trajectories.

### Example: Plot Validation PSNR across experiments
```bash
python plot_experiments.py --base_dir results/train_26-04-26_..._fmdd --metric psnr --output comparison_psnr.png
```

### Example: Plot Train and Validation Loss
```bash
python plot_experiments.py --base_dir results/train_26-04-26_..._fmdd --metric loss --mode both --output comparison_loss.png
```
