import argparse
import json
import os
import random
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CocoSegmentationDataset(Dataset):
    """
    Loads undistorted images listed in split_file (train.txt / val.txt / test.txt)
    and parses COCO-format segmentation annotations exported from CVAT, Label Studio, or Roboflow.
    Categories expected:
      1 -> target_object (the object to measure)
      2 -> reference_object (known size reference object in the same plane)
    """
    def __init__(self, images_dir, coco_json_path, split_txt_path):
        self.images_dir = images_dir
        with open(split_txt_path, "r") as f:
            self.allowed_filenames = set(line.strip() for line in f if line.strip())

        with open(coco_json_path, "r") as f:
            coco = json.load(f)

        # Map category names/IDs cleanly to 1 (target_object) and 2 (reference_object)
        self.cat_map = {}
        for c in coco.get("categories", []):
            cname = c["name"].lower()
            if "ref" in cname or "card" in cname or "coin" in cname or "marker" in cname:
                self.cat_map[c["id"]] = 2
            else:
                self.cat_map[c["id"]] = 1

        self.images_info = [
            img for img in coco["images"]
            if os.path.basename(img["file_name"]) in self.allowed_filenames
        ]
        self.ann_by_img = {}
        for ann in coco["annotations"]:
            img_id = ann["image_id"]
            self.ann_by_img.setdefault(img_id, []).append(ann)

    def __len__(self):
        return len(self.images_info)

    def _poly_to_mask(self, polygons, width, height):
        mask_img = Image.new("L", (width, height), 0)
        for poly in polygons:
            if len(poly) >= 6:
                xy = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
                ImageDraw.Draw(mask_img).polygon(xy, outline=1, fill=1)
        return np.array(mask_img, dtype=np.uint8)

    def __getitem__(self, idx):
        img_meta = self.images_info[idx]
        fname = os.path.basename(img_meta["file_name"])
        img_path = os.path.join(self.images_dir, fname)
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        anns = self.ann_by_img.get(img_meta["id"], [])
        boxes = []
        labels = []
        masks = []

        for ann in anns:
            seg = ann.get("segmentation", [])
            if isinstance(seg, list) and len(seg) > 0:
                mask = self._poly_to_mask(seg, w, h)
            else:
                continue

            pos = np.where(mask > 0)
            if len(pos[0]) == 0:
                continue
            xmin, xmax = float(np.min(pos[1])), float(np.max(pos[1]))
            ymin, ymax = float(np.min(pos[0])), float(np.max(pos[0]))
            if xmax <= xmin or ymax <= ymin:
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.cat_map.get(ann["category_id"], 1))
            masks.append(mask)

        img_tensor = torchvision.transforms.functional.to_tensor(img)

        if len(boxes) == 0:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "masks": torch.zeros((0, h, w), dtype=torch.uint8),
                "image_id": torch.tensor([idx]),
            }
        else:
            target = {
                "boxes": torch.as_tensor(boxes, dtype=torch.float32),
                "labels": torch.as_tensor(labels, dtype=torch.int64),
                "masks": torch.as_tensor(np.stack(masks, axis=0), dtype=torch.uint8),
                "image_id": torch.tensor([idx]),
            }
        return img_tensor, target


def build_maskrcnn_model(num_classes=3):
    # num_classes = 3: 0=background, 1=target_object, 2=reference_object
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn_v2(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    return model


def collate_fn(batch):
    return tuple(zip(*batch))


def train_model(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Mask R-CNN (ResNet50-FPN-v2) on device: {device}")

    train_ds = CocoSegmentationDataset(args.images_dir, args.coco_json, args.train_split)
    val_ds = CocoSegmentationDataset(args.images_dir, args.coco_json, args.val_split)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = build_maskrcnn_model(num_classes=3).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    best_val_loss = float("inf")
    os.makedirs(os.path.dirname(args.out_weights), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            epoch_loss += losses.item()

        avg_train_loss = epoch_loss / max(1, len(train_loader))

        # Validation loss check (model kept in train mode under torch.no_grad() to compute loss dict)
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                val_loss += sum(loss for loss in loss_dict.values()).item()
        avg_val_loss = val_loss / max(1, len(val_loader))

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), args.out_weights)
            print(f"  -> Saved new best checkpoint to {args.out_weights}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Mask R-CNN on custom undistorted dataset.")
    parser.add_argument("--images_dir", type=str, default="dataset/images")
    parser.add_argument("--coco_json", type=str, default="dataset/annotations.json")
    parser.add_argument("--train_split", type=str, default="dataset/train.txt")
    parser.add_argument("--val_split", type=str, default="dataset/val.txt")
    parser.add_argument("--out_weights", type=str, default="models/maskrcnn_best.pth")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_model(args)
