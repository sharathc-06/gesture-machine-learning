"""Transfer-learning training script for the Naruto image classifier.

Loads images from data/naruto_images/<class>/, trains a MobileNetV3-Small
(ImageNet-pretrained) with a replaced classification head, and saves
models/naruto_image_model.pth.

SESSION-AWARE SPLITTING (avoiding the leakage problem from the earlier
landmark-model evaluation)
--------------------------------------------------------------------------
scripts/collect_naruto_images.py names every file
`<class>_<session_id>_<seq>.jpg`. This script parses the session id out of
each filename and splits by (class, session) GROUP rather than by individual
image, using sklearn's GroupShuffleSplit. That means all images from one
sitting go entirely into train, or entirely into val/test - never split
across the boundary - so validation/test accuracy actually estimates
generalization to a *new* session (different lighting/position/day) instead
of just recognizing near-duplicate frames it already trained on.

If a class only has images from a single session (common on your very first
collection run), it CANNOT be split without leakage - there's only one group.
In that case this script falls back to a random per-image split for that
class only, and prints a clear WARNING that its val/test numbers are
leakage-prone and will look better than real-world performance until you
collect a second session for it.
"""
import argparse
import logging
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GroupShuffleSplit

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.image_preprocessing import preprocess_image_file, letterbox_to_square, IMAGENET_MEAN, IMAGENET_STD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_naruto_image_classifier")

CLASS_NAMES = ["tiger", "ox", "hare", "monkey", "horse", "snake", "ram", "boar", "bowl"]
FILENAME_RE = re.compile(r"^(?P<label>[a-z_]+)_(?P<session>\d{8}_\d{6})_(?P<seq>\d+)\.jpg$")


def scan_dataset(data_dir: Path):
    """Return a list of (filepath, label, session_id) for every image found.
    Files that don't match the expected naming convention are skipped with a
    warning rather than crashing the whole run (e.g. a stray non-dataset file
    someone dropped into a class folder)."""
    samples = []
    for cls in CLASS_NAMES:
        cls_dir = data_dir / cls
        if not cls_dir.exists():
            continue
        for f in sorted(cls_dir.glob("*.jpg")):
            m = FILENAME_RE.match(f.name)
            if not m or m.group("label") != cls:
                logger.warning(f"Skipping file with unexpected name: {f}")
                continue
            samples.append((f, cls, m.group("session")))
    return samples


def session_aware_split(samples, test_size=0.15, val_size=0.15, seed=42):
    """Split (path, label, session) samples into train/val/test, grouping by
    (label, session) so an entire session for a class stays on one side of
    the split. Returns (train, val, test) lists and a list of class names
    that had to fall back to a random per-image split (single-session
    classes) for the caller to warn about."""
    by_class = defaultdict(list)
    for s in samples:
        by_class[s[1]].append(s)

    train, val, test = [], [], []
    leaky_classes = []
    rng = np.random.RandomState(seed)

    for cls, cls_samples in by_class.items():
        sessions = sorted(set(s[2] for s in cls_samples))
        if len(sessions) < 3:
            # Not enough distinct sessions to carve out a leakage-free
            # train/val/test split by group - fall back to a random
            # per-image split for this class and flag it.
            leaky_classes.append(cls)
            idx = rng.permutation(len(cls_samples))
            n = len(cls_samples)
            n_test = max(1, int(n * test_size)) if n >= 5 else 0
            n_val = max(1, int(n * val_size)) if n >= 5 else 0
            test_idx = set(idx[:n_test])
            val_idx = set(idx[n_test:n_test + n_val])
            for i, s in enumerate(cls_samples):
                (test if i in test_idx else val if i in val_idx else train).append(s)
            continue

        groups = [s[2] for s in cls_samples]
        gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        trainval_idx, test_idx = next(gss1.split(cls_samples, groups=groups))
        trainval = [cls_samples[i] for i in trainval_idx]
        test.extend(cls_samples[i] for i in test_idx)

        trainval_groups = [s[2] for s in trainval]
        relative_val_size = val_size / (1 - test_size)
        gss2 = GroupShuffleSplit(n_splits=1, test_size=relative_val_size, random_state=seed)
        tr_idx, va_idx = next(gss2.split(trainval, groups=trainval_groups))
        train.extend(trainval[i] for i in tr_idx)
        val.extend(trainval[i] for i in va_idx)

    return train, val, test, leaky_classes


def build_dataset_tensors(samples, input_size, augment: bool):
    """Load and preprocess a list of (path, label, session) samples into a
    single (X, y) tensor pair. Kept simple (whole split in memory at once)
    rather than a streaming DataLoader/Dataset class, since a few thousand
    224x224 images fits comfortably in memory and this keeps the script easy
    to follow; revisit only if the dataset grows far beyond that."""
    import torch
    import torchvision.transforms as T
    from PIL import Image

    label_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}
    aug_transform = None
    if augment:
        # Deliberately NO horizontal flip: several Naruto seals are
        # left/right-hand-position-dependent (e.g. ox: "right hand
        # horizontal, left hand vertical"), so flipping could silently
        # relabel one sign as another's mirror image.
        aug_transform = T.Compose([
            T.RandomRotation(12),
            T.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.9, 1.1)),
            T.ColorJitter(brightness=0.25, contrast=0.25),
        ])

    xs, ys = [], []
    for path, label, _session in samples:
        img = Image.open(path).convert("RGB")
        img = letterbox_to_square(img, input_size)
        if aug_transform is not None:
            img = aug_transform(img)
        tensor = T.functional.to_tensor(img)
        tensor = T.functional.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        xs.append(tensor)
        ys.append(label_to_idx[label])

    if not xs:
        return None, None
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)


def build_model(num_classes: int):
    import torch.nn as nn
    import torchvision.models as models

    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = models.mobilenet_v3_small(weights=weights)
    # mobilenet_v3_small's classifier is Sequential(Linear, Hardswish, Dropout, Linear);
    # replace the final Linear (index 3) to output our class count.
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def set_backbone_trainable(model, trainable: bool):
    for name, param in model.named_parameters():
        if not name.startswith("classifier"):
            param.requires_grad = trainable


def run_epoch(model, X, y, batch_size, optimizer=None, device="cpu"):
    import torch
    import torch.nn.functional as F

    is_train = optimizer is not None
    model.train(is_train)
    n = X.shape[0]
    idx = torch.randperm(n) if is_train else torch.arange(n)
    total_loss, correct = 0.0, 0

    for start in range(0, n, batch_size):
        batch_idx = idx[start:start + batch_size]
        xb, yb = X[batch_idx].to(device), y[batch_idx].to(device)
        if is_train:
            optimizer.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
        total_loss += loss.item() * xb.size(0)
        correct += (logits.argmax(dim=1) == yb).sum().item()

    return total_loss / n, correct / n


def main():
    parser = argparse.ArgumentParser(description="Train the Naruto image classifier via transfer learning")
    parser.add_argument("--data-dir", type=str, default="data/naruto_images")
    parser.add_argument("--model-out", type=str, default="models/naruto_image_model.pth")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs-head", type=int, default=10,
                         help="Epochs training only the new classifier head (backbone frozen)")
    parser.add_argument("--epochs-finetune", type=int, default=5,
                         help="Additional epochs fine-tuning the whole network at a lower LR (0 to skip)")
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-finetune", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    samples = scan_dataset(data_dir)
    if not samples:
        logger.error(f"No images found under {data_dir}. Run scripts/collect_naruto_images.py first.")
        sys.exit(1)

    counts = defaultdict(int)
    for _, label, _ in samples:
        counts[label] += 1
    logger.info(f"Found {len(samples)} images across {len(counts)} classes: {dict(counts)}")
    missing = [c for c in CLASS_NAMES if counts[c] == 0]
    if missing:
        logger.warning(f"No images at all for: {missing}. The model won't be able to recognize these signs "
                        f"until you collect data for them.")

    train, val, test, leaky_classes = session_aware_split(samples)
    if leaky_classes:
        logger.warning(
            f"Classes with images from fewer than 3 sessions ({leaky_classes}) fell back to a random "
            f"per-image split - their val/test accuracy is leakage-prone and will look better than real "
            f"live-webcam performance until you collect at least 2-3 separate sessions for them."
        )
    logger.info(f"Split sizes: train={len(train)} val={len(val)} test={len(test)}")
    if len(val) == 0 or len(test) == 0:
        logger.error("Validation or test split is empty - need more images/sessions per class. Aborting.")
        sys.exit(1)

    logger.info("Loading and preprocessing images (this can take a while for large datasets)...")
    X_train, y_train = build_dataset_tensors(train, args.input_size, augment=True)
    X_val, y_val = build_dataset_tensors(val, args.input_size, augment=False)
    X_test, y_test = build_dataset_tensors(test, args.input_size, augment=False)

    model = build_model(num_classes=len(CLASS_NAMES))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    logger.info(f"Training on device: {device}")

    # --- Stage 1: train only the new classifier head, backbone frozen ---
    set_backbone_trainable(model, trainable=False)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr_head)
    logger.info(f"=== Stage 1: training classifier head ({args.epochs_head} epochs, backbone frozen) ===")
    best_val_acc = -1.0
    best_state = None
    for epoch in range(1, args.epochs_head + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, X_train, y_train, args.batch_size, optimizer, device)
        val_loss, val_acc = run_epoch(model, X_val, y_val, args.batch_size, None, device)
        logger.info(f"[head {epoch}/{args.epochs_head}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                    f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} ({time.time()-t0:.1f}s)")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # --- Stage 2 (optional): fine-tune the whole network at a low LR ---
    if args.epochs_finetune > 0:
        set_backbone_trainable(model, trainable=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_finetune)
        logger.info(f"=== Stage 2: fine-tuning full network ({args.epochs_finetune} epochs, lr={args.lr_finetune}) ===")
        for epoch in range(1, args.epochs_finetune + 1):
            t0 = time.time()
            train_loss, train_acc = run_epoch(model, X_train, y_train, args.batch_size, optimizer, device)
            val_loss, val_acc = run_epoch(model, X_val, y_val, args.batch_size, None, device)
            logger.info(f"[finetune {epoch}/{args.epochs_finetune}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                        f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} ({time.time()-t0:.1f}s)")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    logger.info(f"Best validation accuracy achieved: {best_val_acc:.4f}")
    model.load_state_dict(best_state)

    # --- Final test-set evaluation ---
    model.eval()
    with torch.no_grad():
        logits = model(X_test.to(device))
        y_pred = logits.argmax(dim=1).cpu().numpy()
    y_true = y_test.numpy()
    test_acc = accuracy_score(y_true, y_pred)
    present_classes = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    present_names = [CLASS_NAMES[i] for i in present_classes]
    report = classification_report(y_true, y_pred, labels=present_classes, target_names=present_names, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=present_classes)
    logger.info(f"=== FINAL TEST SET RESULTS ===")
    logger.info(f"Test accuracy: {test_acc:.4f}")
    logger.info(f"Classification report:\n{report}")
    logger.info(f"Confusion matrix (order={present_names}):\n{cm}")

    out_path = Path(args.model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": best_state,
        "classes": CLASS_NAMES,
        "model_name": "mobilenet_v3_small",
        "input_size": args.input_size,
    }, out_path)
    logger.info(f"Saved best model (val_acc={best_val_acc:.4f}, test_acc={test_acc:.4f}) to {out_path}")


if __name__ == "__main__":
    main()
