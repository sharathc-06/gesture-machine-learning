"""Shared image preprocessing for the Naruto image classifier.

Used by BOTH scripts/train_naruto_image_classifier.py and
src/naruto_image_classifier.py so training and real-time inference can never
drift apart into two subtly different preprocessing pipelines - a classic
source of "trains fine, performs badly live" bugs.

Preprocessing: letterbox resize (preserve aspect ratio, pad with black to a
square) to `input_size`, then ImageNet normalization. Letterboxing rather than
a center-crop or a naive aspect-ratio-distorting resize was chosen specifically
because a plain resize/crop risks cutting off or squishing hand geometry,
which the project's Naruto hand signs depend on.
"""
from typing import Tuple

import numpy as np
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def letterbox_to_square(image: Image.Image, size: int) -> Image.Image:
    """Resize `image` to fit within `size`x`size` preserving aspect ratio,
    padding the remainder with black. Never crops - the full original frame
    content is always preserved somewhere in the output."""
    w, h = image.size
    scale = size / max(w, h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = image.resize((new_w, new_h), Image.BILINEAR)

    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    paste_x = (size - new_w) // 2
    paste_y = (size - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))
    return canvas


def bgr_ndarray_to_pil(frame_bgr: np.ndarray) -> Image.Image:
    """Convert an OpenCV BGR ndarray frame to a PIL RGB Image."""
    rgb = frame_bgr[:, :, ::-1]
    return Image.fromarray(rgb)


def preprocess_for_model(frame_bgr: np.ndarray, input_size: int):
    """Full preprocessing pipeline for one OpenCV BGR frame -> normalized
    torch tensor of shape (3, input_size, input_size), ready to batch with
    unsqueeze(0). Imports torch lazily so this module (and the letterbox
    logic above) can be exercised/tested without torch installed."""
    import torch
    import torchvision.transforms.functional as TF

    pil_img = bgr_ndarray_to_pil(frame_bgr)
    letterboxed = letterbox_to_square(pil_img, input_size)
    tensor = TF.to_tensor(letterboxed)
    tensor = TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return tensor


def preprocess_image_file(path: str, input_size: int):
    """Same pipeline as preprocess_for_model, but loading from a file path
    (used by the training script, which reads saved dataset images rather
    than live BGR frames)."""
    import torch
    import torchvision.transforms.functional as TF

    pil_img = Image.open(path).convert("RGB")
    letterboxed = letterbox_to_square(pil_img, input_size)
    tensor = TF.to_tensor(letterboxed)
    tensor = TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return tensor
