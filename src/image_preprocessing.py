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
from typing import List, Tuple

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


# Default padding, as a fraction of the union bounding box's own width/height,
# added on EACH side. E.g. 0.3 means 30% of the box width added to the left
# AND 30% added to the right (so the crop is up to ~1.6x wider than the raw
# box), same fraction applied to height. Kept as a single named constant so
# it's easy to find and tune from one place while eyeballing the visual test,
# rather than a magic number buried in a call site.
DEFAULT_HAND_CROP_PADDING = 0.3


def crop_hand_region(
    frame_bgr: np.ndarray,
    bbox: Tuple[float, float, float, float],
    padding: float = DEFAULT_HAND_CROP_PADDING,
) -> np.ndarray:
    """Crop `frame_bgr` to the (padded, boundary-clipped) region described by
    `bbox`.

    Args:
        frame_bgr: full camera frame (already flipped, same convention as
            every other frame in this codebase), as an OpenCV BGR ndarray.
        bbox: (x_min, y_min, x_max, y_max) in absolute pixel coordinates -
            e.g. straight from HandDetector.get_hand_bboxes(), or the union
            of several such boxes for a two-hand sign (union math is the
            caller's job, this function just crops one box).
        padding: fraction of the box's own width/height to pad on EACH side.
            See DEFAULT_HAND_CROP_PADDING above for the exact semantics.

    Returns:
        The cropped region as a new BGR ndarray (a view/copy of frame_bgr,
        not resized/letterboxed - call letterbox_to_square() separately on
        a PIL conversion of this, same as the existing pipeline does for
        the full frame today).

    Padding is applied then the result is clamped to the frame's actual
    boundaries, so a hand near the frame edge never raises or wraps - it
    just gets less padding on that side rather than an out-of-bounds crop.
    """
    frame_h, frame_w = frame_bgr.shape[:2]
    x_min, y_min, x_max, y_max = bbox

    box_w = max(1.0, x_max - x_min)
    box_h = max(1.0, y_max - y_min)
    pad_x = box_w * padding
    pad_y = box_h * padding

    x1 = int(round(x_min - pad_x))
    y1 = int(round(y_min - pad_y))
    x2 = int(round(x_max + pad_x))
    y2 = int(round(y_max + pad_y))

    # Clip to frame boundaries - never index outside the array, and never
    # silently wrap/error on a hand near an edge.
    x1 = max(0, min(x1, frame_w - 1))
    y1 = max(0, min(y1, frame_h - 1))
    x2 = max(x1 + 1, min(x2, frame_w))
    y2 = max(y1 + 1, min(y2, frame_h))

    return frame_bgr[y1:y2, x1:x2]


def union_bbox(bboxes: List[Tuple[float, float, float, float]]) -> Tuple[float, float, float, float]:
    """Combine multiple (x_min, y_min, x_max, y_max) boxes into one box that
    contains all of them - used for two-hand signs, where the CNN should see
    both hands in a single crop. Caller must pass a non-empty list."""
    xs_min = [b[0] for b in bboxes]
    ys_min = [b[1] for b in bboxes]
    xs_max = [b[2] for b in bboxes]
    ys_max = [b[3] for b in bboxes]
    return (min(xs_min), min(ys_min), max(xs_max), max(ys_max))


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
