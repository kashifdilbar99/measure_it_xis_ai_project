import argparse
import glob
import os
import random


def create_splits(images_dir, out_dir, seed=42):
    random.seed(seed)
    files = sorted([
        os.path.basename(p)
        for p in (
            glob.glob(os.path.join(images_dir, "*.jpg"))
            + glob.glob(os.path.join(images_dir, "*.png"))
            + glob.glob(os.path.join(images_dir, "*.jpeg"))
        )
    ])
    if len(files) < 70:
        print(f"[WARNING] Found {len(files)} images. Assessment requires >= 70 images!")

    random.shuffle(files)
    n = len(files)
    n_train = int(round(n * 0.70))
    n_val = int(round(n * 0.20))

    train_files = sorted(files[:n_train])
    val_files = sorted(files[n_train:n_train + n_val])
    test_files = sorted(files[n_train + n_val:])

    os.makedirs(out_dir, exist_ok=True)
    for name, subset in [("train.txt", train_files), ("val.txt", val_files), ("test.txt", test_files)]:
        with open(os.path.join(out_dir, name), "w") as f:
            for item in subset:
                f.write(f"{item}\n")

    print(f"Total: {n} | Train (70%): {len(train_files)} | Val (20%): {len(val_files)} | Test (10%): {len(test_files)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create deterministic 70/20/10 train/val/test split lists.")
    parser.add_argument("--images_dir", type=str, default="dataset/images")
    parser.add_argument("--out_dir", type=str, default="dataset")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    create_splits(args.images_dir, args.out_dir, args.seed)
