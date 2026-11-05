import argparse
import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "inference"))
from measure import run_measurement


def evaluate_calliper_vs_system(args):
    df = pd.read_csv(args.ground_truth_csv)
    rows = []

    for _, row in df.iterrows():
        img_path = row["image_path"]
        true_w = float(row["calliper_width_mm"])
        true_h = float(row["calliper_height_mm"])

        # 1. Undistorted measurement (Item 5)
        res_undist = run_measurement(
            img_path, args.calib, args.weights, args.ref_w_mm, args.ref_h_mm,
            out_path=f"measurement/overlays/undist_{os.path.basename(img_path)}",
            skip_undistort=False, min_conf=args.min_conf
        )

        # 2. Raw distorted measurement (Item 6: Proof undistortion matters)
        res_raw = run_measurement(
            img_path, args.calib, args.weights, args.ref_w_mm, args.ref_h_mm,
            out_path=f"measurement/overlays/raw_{os.path.basename(img_path)}",
            skip_undistort=True, min_conf=args.min_conf
        )

        if res_undist.get("status") == "OK" and res_raw.get("status") == "OK":
            uw, uh = res_undist["width_mm"], res_undist["height_mm"]
            rw, rh = res_raw["width_mm"], res_raw["height_mm"]

            u_err_w = abs(uw - true_w)
            u_err_h = abs(uh - true_h)
            u_mpe = 0.5 * ((u_err_w / true_w) + (u_err_h / true_h)) * 100.0

            r_err_w = abs(rw - true_w)
            r_err_h = abs(rh - true_h)
            r_mpe = 0.5 * ((r_err_w / true_w) + (r_err_h / true_h)) * 100.0

            rows.append({
                "instance": os.path.basename(img_path),
                "true_w_mm": true_w,
                "true_h_mm": true_h,
                "undist_w_mm": uw,
                "undist_h_mm": uh,
                "undist_err_w_mm": round(u_err_w, 3),
                "undist_err_h_mm": round(u_err_h, 3),
                "undist_mpe_pct": round(u_mpe, 3),
                "raw_w_mm": rw,
                "raw_h_mm": rh,
                "raw_err_w_mm": round(r_err_w, 3),
                "raw_err_h_mm": round(r_err_h, 3),
                "raw_mpe_pct": round(r_mpe, 3),
            })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out_csv, index=False)
    print("\n=== PER-INSTANCE ACCURACY & UNDISTORTION COMPARISON (ITEMS 5 & 6) ===")
    print(out_df.to_string(index=False))

    if len(out_df) > 0:
        for prefix, label in [("undist", "UNDISTORTED"), ("raw", "RAW (DISTORTED)")]:
            all_errs = np.concatenate([out_df[f"{prefix}_err_w_mm"].values, out_df[f"{prefix}_err_h_mm"].values])
            print(f"\n--- {label} SUMMARY ---")
            print(f"Mean Absolute Error (MAE) : {np.mean(all_errs):.3f} mm")
            print(f"Median Absolute Error     : {np.median(all_errs):.3f} mm")
            print(f"Worst-Case Error          : {np.max(all_errs):.3f} mm")
            print(f"Mean Percentage Error     : {out_df[f'{prefix}_mpe_pct'].mean():.3f} %")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate system accuracy against calliper on >=10 instances.")
    parser.add_argument("--ground_truth_csv", type=str, default="measurement/ground_truth.csv")
    parser.add_argument("--calib", type=str, default="calibration/camera_params.json")
    parser.add_argument("--weights", type=str, default="models/maskrcnn_best.pth")
    parser.add_argument("--ref_w_mm", type=float, default=85.60)
    parser.add_argument("--ref_h_mm", type=float, default=53.98)
    parser.add_argument("--min_conf", type=float, default=0.75)
    parser.add_argument("--out_csv", type=str, default="measurement/results_comparison.csv")
    args = parser.parse_args()
    evaluate_calliper_vs_system(args)
