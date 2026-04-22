import os
import argparse
import numpy as np
import torch
import tifffile
import imageio.v3 as iio
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from skimage.metrics import structural_similarity as ssim_func
import csv

def load_image(path):
    """Loads an image and normalizes it to [0, 1] if needed, returns a numpy array."""
    img = iio.imread(path)
    if img.ndim == 3 and img.shape[-1] == 3:
        # Convert RGB to grayscale (standard for these metrics in this project)
        img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
    
    # Ensure it's float [0, 1] for metric stability
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    else:
        # If already float, check range. If it's 0-255, scale down.
        if img.max() > 2.0:
            img = img.astype(np.float32) / 255.0
            
    return img

def compute_metrics(denoised_path, gt_path):
    """Computes PSNR and SSIM between two images."""
    denoised = load_image(denoised_path)
    gt = load_image(gt_path)
    
    # Ensure dimensions match
    if denoised.shape != gt.shape:
        # Simple crop to match if tiny difference due to divisibility
        h = min(denoised.shape[0], gt.shape[0])
        w = min(denoised.shape[1], gt.shape[1])
        denoised = denoised[:h, :w]
        gt = gt[:h, :w]

    psnr_val = psnr_func(gt, denoised, data_range=1.0)
    ssim_val = ssim_func(gt, denoised, data_range=1.0)
    
    return psnr_val, ssim_val

def main():
    parser = argparse.ArgumentParser(description="Compute PSNR and SSIM metrics for denoising experiments.")
    parser.add_argument("--results_dir", type=str, required=True, help="Path to the experiment folder (containing lr_... subdirs).")
    parser.add_argument("--gt", type=str, required=True, help="Path to GT image or template (e.g., path/to/gt_%03d.png).")
    parser.add_argument("--first", type=int, default=0, help="First frame index (for sequential data).")
    parser.add_argument("--last", type=int, default=0, help="Last frame index (for sequential data).")
    parser.add_argument("--output_csv", type=str, default="experiment_metrics.csv", help="Name of the output CSV file.")
    parser.add_argument("--save_diff", action="store_true", help="Save absolute difference maps as images.")
    
    args = parser.parse_args()

    # Find all sub-experiments (folders with 'lr_' in their name)
    sub_experiments = [d for d in os.listdir(args.results_dir) if os.path.isdir(os.path.join(args.results_dir, d)) and "lr_" in d]
    
    if not sub_experiments:
        # If no subfolders found, maybe results_dir IS the experiment folder
        sub_experiments = ["."]

    all_results = []

    for exp in sorted(sub_experiments):
        print(f"\n[*] Evaluating experiment: {exp}")
        exp_path = os.path.join(args.results_dir, exp, "best_model")
        if not os.path.exists(exp_path):
            exp_path = os.path.join(args.results_dir, exp)
            
        is_gt_template = "%" in args.gt
        scan_range = range(args.first, args.last + 1) if is_gt_template else [args.first]

        exp_psnrs, exp_ssims = [], []

        for i in scan_range:
            found_file = None
            if os.path.exists(exp_path):
                for f in os.listdir(exp_path):
                    if f.startswith("test_output") and f.endswith(".tif") and f"{i:03d}" in f:
                        found_file = os.path.join(exp_path, f)
                        break
            
            if found_file:
                cur_gt = args.gt % i if is_gt_template else args.gt
                if os.path.exists(cur_gt):
                    try:
                        denoised = load_image(found_file)
                        gt = load_image(cur_gt)
                        
                        # Handle size mismatch
                        h, w = min(denoised.shape[0], gt.shape[0]), min(denoised.shape[1], gt.shape[1])
                        denoised, gt = denoised[:h, :w], gt[:h, :w]

                        p_val = psnr_func(gt, denoised, data_range=1.0)
                        s_val = ssim_func(gt, denoised, data_range=1.0)
                        
                        exp_psnrs.append(p_val)
                        exp_ssims.append(s_val)
                        print(f"  Frame {i:03d} [{os.path.basename(found_file)}]: PSNR = {p_val:.2f} dB, SSIM = {s_val:.4f}")

                        if args.save_diff:
                            diff = np.abs(gt - denoised)
                            # Normalize diff for better visualization if small
                            diff_norm = (diff / (diff.max() + 1e-6) * 65535).astype(np.uint16)
                            diff_path = os.path.join(exp_path, f"diff_{i:03d}.tif")
                            tifffile.imwrite(diff_path, diff_norm)
                    except Exception as e:
                        print(f"  [!] Error processing {found_file}: {e}")
                else:
                    print(f"  [!] Missing GT: {cur_gt}")

        if exp_psnrs:
            avg_psnr, avg_ssim = np.mean(exp_psnrs), np.mean(exp_ssims)
            print(f"--- Summary for {exp}: AVG PSNR = {avg_psnr:.2f} dB, AVG SSIM = {avg_ssim:.4f}")
            all_results.append({
                "Experiment": exp,
                "Avg_PSNR": avg_psnr,
                "Avg_SSIM": avg_ssim,
                "Num_Frames": len(exp_psnrs)
            })

    if all_results:
        # Save to CSV
        output_file = os.path.join(args.results_dir, args.output_csv)
        keys = all_results[0].keys()
        with open(output_file, 'w', newline='') as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(all_results)
            
        print(f"\n[+] Global results saved to {output_file}")
        
        # Print summary table manually
        header = f"{'Experiment':<50} | {'Avg PSNR':<10} | {'Avg SSIM':<10} | {'Frames':<7}"
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        for res in all_results:
            print(f"{res['Experiment']:<50} | {res['Avg_PSNR']:<10.2f} | {res['Avg_SSIM']:<10.4f} | {res['Num_Frames']:<7}")
        print("-" * len(header))
    else:
        print("\n[!] No results found to evaluate.")

if __name__ == "__main__":
    main()
