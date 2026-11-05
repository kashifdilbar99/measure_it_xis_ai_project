import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from train import CocoSegmentationDataset, build_maskrcnn_model, collate_fn


def compute_mask_iou(mask1, mask2):
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return float(inter / union) if union > 0 else 0.0


@torch.no_grad()
def evaluate_test_split(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_ds = CocoSegmentationDataset(args.images_dir, args.coco_json, args.test_split)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = build_maskrcnn_model(num_classes=3).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    iou_thresholds = np.arange(0.50, 1.00, 0.05)
    ious_collected = []
    tp_at_50, fp_at_50, fn_at_50 = 0, 0, 0
    ap_per_thresh = {round(t, 2): [] for t in iou_thresholds}

    for images, targets in test_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)[0]

        gt_masks = targets[0]["masks"].numpy() > 0
        gt_labels = targets[0]["labels"].numpy()

        scores = outputs["scores"].cpu().numpy()
        keep = scores >= args.conf_thresh
        pred_masks = (outputs["masks"][keep, 0].cpu().numpy() >= 0.5)
        pred_labels = outputs["labels"][keep].cpu().numpy()

        matched_gt = set()
        for p_idx in range(len(pred_masks)):
            best_iou = 0.0
            best_g = -1
            for g_idx in range(len(gt_masks)):
                if g_idx in matched_gt or pred_labels[p_idx] != gt_labels[g_idx]:
                    continue
                iou = compute_mask_iou(pred_masks[p_idx], gt_masks[g_idx])
                if iou > best_iou:
                    best_iou = iou
                    best_g = g_idx

            if best_iou >= 0.50 and best_g != -1:
                tp_at_50 += 1
                matched_gt.add(best_g)
                ious_collected.append(best_iou)
            else:
                fp_at_50 += 1

            for t in iou_thresholds:
                ap_per_thresh[round(t, 2)].append(1.0 if best_iou >= t else 0.0)

        fn_at_50 += max(0, len(gt_masks) - len(matched_gt))

    mean_iou = float(np.mean(ious_collected)) if ious_collected else 0.0
    precision = tp_at_50 / max(1, tp_at_50 + fp_at_50)
    recall = tp_at_50 / max(1, tp_at_50 + fn_at_50)
    map_50 = float(np.mean(ap_per_thresh[0.5])) if ap_per_thresh[0.5] else 0.0
    map_50_95 = float(np.mean([np.mean(v) if v else 0.0 for v in ap_per_thresh.values()]))

    print("=== TEST SPLIT EVALUATION METRICS ===")
    print(f"Test Images Evaluated : {len(test_ds)}")
    print(f"Mean Mask IoU         : {mean_iou:.4f}")
    print(f"mAP@0.5               : {map_50:.4f}")
    print(f"mAP@0.5:0.95          : {map_50_95:.4f}")
    print(f"Precision (@0.5 IoU)  : {precision:.4f}")
    print(f"Recall (@0.5 IoU)     : {recall:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained Mask R-CNN on locked test split.")
    parser.add_argument("--images_dir", type=str, default="dataset/images")
    parser.add_argument("--coco_json", type=str, default="dataset/annotations.json")
    parser.add_argument("--test_split", type=str, default="dataset/test.txt")
    parser.add_argument("--weights", type=str, default="models/maskrcnn_best.pth")
    parser.add_argument("--conf_thresh", type=float, default=0.5)
    args = parser.parse_args()
    evaluate_test_split(args)
