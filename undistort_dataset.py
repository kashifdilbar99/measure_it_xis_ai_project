import argparse
import glob
import json
import os
import cv2
import numpy as np


def load_calibration(calib_file):
    with open(calib_file, "r") as f:
        data = json.load(f)
    camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.array(data["dist_coeffs"], dtype=np.float64)
    return camera_matrix, dist_coeffs


def undistort_image(img, camera_matrix, dist_coeffs):
    h, w = img.shape[:2]
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 0, (w, h))
    undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_mtx)
    return undistorted


def batch_undistort(raw_dir, out_dir, calib_file):
    camera_matrix, dist_coeffs = load_calibration(calib_file)
    os.makedirs(out_dir, exist_ok=True)
    paths = sorted(
        glob.glob(os.path.join(raw_dir, "*.jpg"))
        + glob.glob(os.path.join(raw_dir, "*.png"))
        + glob.glob(os.path.join(raw_dir, "*.jpeg"))
    )
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        undist = undistort_image(img, camera_matrix, dist_coeffs)
        out_path = os.path.join(out_dir, os.path.basename(p))
        cv2.imwrite(out_path, undist)
    print(f"Undistorted {len(paths)} images from {raw_dir} -> {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch undistort raw dataset images.")
    parser.add_argument("--raw_dir", type=str, default="dataset/raw_images")
    parser.add_argument("--out_dir", type=str, default="dataset/images")
    parser.add_argument("--calib", type=str, default="calibration/camera_params.json")
    args = parser.parse_args()
    batch_undistort(args.raw_dir, args.out_dir, args.calib)
