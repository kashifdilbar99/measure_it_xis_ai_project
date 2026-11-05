import argparse
import glob
import json
import os
import cv2
import numpy as np


def calibrate_camera(images_dir, pattern_cols, pattern_rows, square_size_mm, output_file):
    # Prepare 3D object points in real-world millimetres: (0,0,0), (1*sq,0,0), ...
    objp = np.zeros((pattern_rows * pattern_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_cols, 0:pattern_rows].T.reshape(-1, 2)
    objp *= square_size_mm

    objpoints = []  # 3D points in real world space (mm)
    imgpoints = []  # 2D points in image plane (pixels)

    image_paths = sorted(
        glob.glob(os.path.join(images_dir, "*.jpg"))
        + glob.glob(os.path.join(images_dir, "*.png"))
        + glob.glob(os.path.join(images_dir, "*.jpeg"))
    )

    if len(image_paths) < 20:
        print(f"[WARNING] Found {len(image_paths)} images. Assessment requires >= 20 images!")

    img_shape = None
    valid_images = []

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_shape is None:
            img_shape = gray.shape[::-1]

        ret, corners = cv2.findChessboardCorners(
            gray,
            (pattern_cols, pattern_rows),
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            valid_images.append(os.path.basename(path))
        else:
            print(f"[SKIP] Checkerboard corners not detected in: {os.path.basename(path)}")

    if len(valid_images) == 0:
        raise RuntimeError("No valid checkerboard images found. Check --cols and --rows inner corner counts.")

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None
    )

    # Compute exact per-image and mean reprojection error in pixels
    total_error = 0.0
    per_image_errors = {}
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
        per_image_errors[valid_images[i]] = round(float(error), 5)

    mean_reproj_error = total_error / len(objpoints)

    calib_data = {
        "image_width": img_shape[0],
        "image_height": img_shape[1],
        "pattern_inner_cols": pattern_cols,
        "pattern_inner_rows": pattern_rows,
        "square_size_mm": square_size_mm,
        "num_valid_images": len(valid_images),
        "rms_calibrate_error": float(ret),
        "mean_reprojection_error_px": round(float(mean_reproj_error), 5),
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.tolist(),
        "per_image_reprojection_error_px": per_image_errors,
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(calib_data, f, indent=2)

    print("=== CAMERA CALIBRATION COMPLETE ===")
    print(f"Valid Checkerboard Images : {len(valid_images)}")
    print(f"Image Resolution          : {img_shape[0]} x {img_shape[1]}")
    print(f"Mean Reprojection Error   : {mean_reproj_error:.5f} pixels")
    print(f"Saved Parameters To       : {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate camera from checkerboard images.")
    parser.add_argument("--images_dir", type=str, default="calibration/images", help="Folder with >=20 checkerboard photos")
    parser.add_argument("--cols", type=int, default=9, help="Inner corners along width (default 9)")
    parser.add_argument("--rows", type=int, default=7, help="Inner corners along height (default 7)")
    parser.add_argument("--square_mm", type=float, default=24.5, help="Measured size of one checkerboard square in mm")
    parser.add_argument("--out", type=str, default="calibration/camera_params.json", help="Output JSON file")
    args = parser.parse_args()

    calibrate_camera(args.images_dir, args.cols, args.rows, args.square_mm, args.out)
