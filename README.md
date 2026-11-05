# Measure It — Camera-Calibrated Instance Segmentation & Metric Measurement

End-to-end computer vision system that undistorts camera images using OpenCV checkerboard calibration, segments a target physical object and a coplanar reference object using a fine-tuned **Mask R-CNN (`maskrcnn_resnet50_fpn_v2`)** model, and converts pixel contours into real-world millimetres validated against a digital calliper.

---

## 1. Hardware & Environment
- **CPU:** `<YOUR_CPU_MODEL>`
- **GPU:** `<YOUR_GPU_MODEL>`
- **RAM:** `<YOUR_RAM_SIZE>`
- **OS:** `<YOUR_OS>`
- **Camera Model:** `<YOUR_PHONE_MODEL>` (Focus, 1x zoom, and resolution locked prior to calibration)
- **Image Resolution:** `1920 x 1080` (update if different)

---

## 2. Dataset Summary
- **Target Object:** `<YOUR_OBJECT_NAME>` (Approx. `<W> mm x <H> mm`, chosen for rigid edges that allow repeatable calliper ground-truth verification)
- **Reference Object:** Standard PVC/ID Card (`85.60 mm x 53.98 mm` verified by calliper, placed flat on the same plane)
- **Total Captured Images:** 75 own images (taken with the calibrated camera and undistorted)
- **Fixed Dataset Split (70/20/10):**
  - Train: 52 images (`dataset/train.txt`)
  - Validation: 15 images (`dataset/val.txt`)
  - Test: 8 images (`dataset/test.txt`)
- **Labelling Tool:** CVAT / Label Studio / Roboflow Annotation Tool (exported in COCO Segmentation JSON format to `dataset/annotations.json`)

---

## 3. Model Weights & Reproducibility
- **Architecture:** TorchVision `maskrcnn_resnet50_fpn_v2` (2 foreground classes: `1=target_object`, `2=reference_object`)
- **Random Seed:** `42` (pinned across Python, NumPy, and PyTorch)
- **Weights Link / Release Asset:** `<PASTE_GITHUB_RELEASE_ASSET_LINK_OR_PATH>`
- **Model Weights SHA-256 Hash:**
  ```text
  <PASTE_SHA256_HASH_FROM_sha256sum_models/maskrcnn_best.pth>
  ```

---

## 4. Setup & Step-by-Step Execution

### Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Stage 1: Camera Calibration (Item 1)
Place 20+ checkerboard images in `calibration/images/` and run:
```bash
python calibration/calibrate.py --images_dir calibration/images --cols 9 --rows 7 --square_mm 24.5 --out calibration/camera_params.json
```

### Stage 2: Undistort Dataset & Lock 70/20/10 Split (Item 2)
```bash
python calibration/undistort_dataset.py --raw_dir dataset/raw_images --out_dir dataset/images --calib calibration/camera_params.json
python dataset/create_splits.py --images_dir dataset/images --out_dir dataset --seed 42
```

### Stage 3: Train & Evaluate Segmentation Model (Item 3)
```bash
python models/train.py --images_dir dataset/images --coco_json dataset/annotations.json -- epochs 20
python models/evaluate.py --images_dir dataset/images --coco_json dataset/annotations.json --test_split dataset/test.txt --weights models/maskrcnn_best.pth
```

### Stage 4: One Command — Image to Millimetres (Items 4, 7, 9)
```bash
python inference/measure.py --image dataset/raw_images/test_01.jpg --calib calibration/camera_params.json --weights models/maskrcnn_best.pth --ref_w_mm 85.60 --ref_h_mm 53.98 --out inference/output_overlay.jpg
```
To benchmark inference speed across 5 runs (median & spread):
```bash
python inference/measure.py --image dataset/raw_images/test_01.jpg --benchmark 5
```

### Stage 5: Accuracy Against Calliper & Raw vs. Undistorted Proof (Items 5 & 6)
```bash
python measurement/evaluate_accuracy.py --ground_truth_csv measurement/ground_truth.csv --out_csv measurement/results_comparison.csv
```

---

## 5. Core Documentation
- [Execution Plan & Decision Record](docs/Plan.md)
- [Measurable Objectives & Results](docs/Objectives.md)
- [Definition of Done Checklist](docs/Definition_of_Done.md)
