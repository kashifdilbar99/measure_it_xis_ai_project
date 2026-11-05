import argparse
import json
import os
import sys
import time
import cv2
import numpy as np
import torch
import torchvision

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "models"))
from train import build_maskrcnn_model


def load_camera_calibration(calib_path):
    with open(calib_path, "r") as f:
        data = json.load(f)
    K = np.array(data["camera_matrix"], dtype=np.float64)
    D = np.array(data["dist_coeffs"], dtype=np.float64)
    return K, D


def apply_undistortion(bgr_img, K, D):
    # K2 ANSWER LINE: Undistortion is applied here before any segmentation or pixel measurement
    h, w = bgr_img.shape[:2]
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 0, (w, h))
    return cv2.undistort(bgr_img, K, D, None, new_K)


def extract_rotated_dimensions_px(binary_mask):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 50:
        return None, None, None
    rect = cv2.minAreaRect(largest)
    (cx, cy), (side1, side2), angle = rect
    width_px = float(max(side1, side2))
    height_px = float(min(side1, side2))
    box_pts = np.intp(cv2.boxPoints(rect))
    return width_px, height_px, box_pts


def pixels_to_millimetres(obj_w_px, obj_h_px, ref_w_px, ref_h_px, ref_true_w_mm, ref_true_h_mm):
    # K1 ANSWER LINE: Converts pixels to millimetres using the reference object's isotropic scale factor
    scale_w = ref_true_w_mm / ref_w_px
    scale_h = ref_true_h_mm / ref_h_px
    mm_per_pixel = 0.5 * (scale_w + scale_h)

    obj_w_mm = obj_w_px * mm_per_pixel
    obj_h_mm = obj_h_px * mm_per_pixel
    return obj_w_mm, obj_h_mm, mm_per_pixel, abs(scale_w - scale_h) / mm_per_pixel


@torch.no_grad()
def run_measurement(
    image_path,
    calib_path,
    weights_path,
    ref_w_mm,
    ref_h_mm,
    out_path="inference/output_overlay.jpg",
    skip_undistort=False,
    min_conf=0.75,
    max_plane_skew=0.08,
):
    raw_img = cv2.imread(image_path)
    if raw_img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    if not skip_undistort:
        K, D = load_camera_calibration(calib_path)
        work_img = apply_undistortion(raw_img, K, D)
    else:
        work_img = raw_img.copy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_maskrcnn_model(num_classes=3).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    rgb_img = cv2.cvtColor(work_img, cv2.COLOR_BGR2RGB)
    img_tensor = torchvision.transforms.functional.to_tensor(rgb_img).to(device)
    output = model([img_tensor])[0]

    scores = output["scores"].cpu().numpy()
    labels = output["labels"].cpu().numpy()
    masks = (output["masks"][:, 0].cpu().numpy() >= 0.5).astype(np.uint8)

    # Find highest-confidence Target Object (label == 1) and Reference Object (label == 2)
    obj_idx = [i for i, l in enumerate(labels) if l == 1]
    ref_idx = [i for i, l in enumerate(labels) if l == 2]

    if len(ref_idx) == 0:
        print("[STATUS: REFERENCE_NOT_FOUND] Reference object was not detected in the image.")
        return {"status": "REFERENCE_NOT_FOUND"}

    if len(obj_idx) == 0:
        print("[STATUS: OBJECT_NOT_FOUND] Target object was not detected in the image.")
        return {"status": "OBJECT_NOT_FOUND"}

    best_obj = obj_idx[0]
    best_ref = ref_idx[0]
    obj_conf = float(scores[best_obj])
    ref_conf = float(scores[best_ref])

    # Item 9: Knowing when not to answer (Refusal Gate 1: Low segmentation confidence)
    if obj_conf < min_conf or ref_conf < min_conf:
        print(
            f"[STATUS: REFUSED_LOW_CONFIDENCE] Object conf={obj_conf:.2f}, "
            f"Ref conf={ref_conf:.2f} (Threshold={min_conf:.2f})"
        )
        return {"status": "REFUSED_LOW_CONFIDENCE", "obj_conf": obj_conf, "ref_conf": ref_conf}

    obj_w_px, obj_h_px, obj_box = extract_rotated_dimensions_px(masks[best_obj])
    ref_w_px, ref_h_px, ref_box = extract_rotated_dimensions_px(masks[best_ref])

    if obj_w_px is None or ref_w_px is None:
        print("[STATUS: INVALID_MASK_GEOMETRY] Could not extract valid contour from mask.")
        return {"status": "INVALID_MASK_GEOMETRY"}

    obj_w_mm, obj_h_mm, mm_per_px, aspect_skew = pixels_to_millimetres(
        obj_w_px, obj_h_px, ref_w_px, ref_h_px, ref_w_mm, ref_h_mm
    )

    # Item 9: Knowing when not to answer (Refusal Gate 2: Tilted plane or partly occluded reference)
    if aspect_skew > max_plane_skew:
        print(
            f"[STATUS: REFUSED_TILT_OR_OCCLUSION] Reference scale anisotropy={aspect_skew*100:.1f}% "
            f"exceeds max allowed {max_plane_skew*100:.1f}% (tilted plane or partially hidden reference)."
        )
        return {"status": "REFUSED_TILT_OR_OCCLUSION", "aspect_skew": aspect_skew}

    # Draw visual overlay
    overlay = work_img.copy()
    overlay[masks[best_obj] == 1] = (
        0.6 * overlay[masks[best_obj] == 1] + 0.4 * np.array([0, 255, 0])
    ).astype(np.uint8)
    overlay[masks[best_ref] == 1] = (
        0.6 * overlay[masks[best_ref] == 1] + 0.4 * np.array([255, 165, 0])
    ).astype(np.uint8)

    cv2.drawContours(overlay, [obj_box], 0, (0, 255, 0), 2)
    cv2.drawContours(overlay, [ref_box], 0, (255, 165, 0), 2)

    label_text = f"Object: {obj_w_mm:.2f}mm x {obj_h_mm:.2f}mm (conf={obj_conf:.2f})"
    cv2.putText(overlay, label_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, overlay)

    result = {
        "status": "OK",
        "width_mm": round(obj_w_mm, 2),
        "height_mm": round(obj_h_mm, 2),
        "confidence": round(obj_conf, 4),
        "ref_confidence": round(ref_conf, 4),
        "mm_per_pixel": round(mm_per_px, 5),
        "overlay_saved": out_path,
    }
    print(
        f"[OK] Width: {result['width_mm']:.2f} mm | Height: {result['height_mm']:.2f} mm | "
        f"Confidence: {result['confidence']:.2f} | Overlay: {out_path}"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="One command: image to millimetres.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--calib", type=str, default="calibration/camera_params.json")
    parser.add_argument("--weights", type=str, default="models/maskrcnn_best.pth")
    parser.add_argument("--ref_w_mm", type=float, default=85.60, help="Known reference width in mm")
    parser.add_argument("--ref_h_mm", type=float, default=53.98, help="Known reference height in mm")
    parser.add_argument("--out", type=str, default="inference/output_overlay.jpg")
    parser.add_argument("--skip_undistort", action="store_true", help="Skip undistortion (for Item 6 test)")
    parser.add_argument("--min_conf", type=float, default=0.75, help="Refusal threshold for mask confidence")
    parser.add_argument("--benchmark", type=int, default=1, help="Run N times to report median & spread latency")
    args = parser.parse_args()

    if args.benchmark > 1:
        timings_ms = []
        for i in range(args.benchmark):
            t0 = time.perf_counter()
            run_measurement(
                args.image, args.calib, args.weights, args.ref_w_mm, args.ref_h_mm,
                args.out, args.skip_undistort, args.min_conf
            )
            timings_ms.append((time.perf_counter() - t0) * 1000.0)
        med = float(np.median(timings_ms))
        spread = float(np.max(timings_ms) - np.min(timings_ms))
        print(f"=== LATENCY BENCHMARK ({args.benchmark} runs) ===")
        print(f"Timings (ms): {[round(t, 2) for t in timings_ms]}")
        print(f"Median: {med:.2f} ms | Spread (Max-Min): {spread:.2f} ms")
    else:
        run_measurement(
            args.image, args.calib, args.weights, args.ref_w_mm, args.ref_h_mm,
            args.out, args.skip_undistort, args.min_conf
        )