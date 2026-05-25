from pycocotools import mask as maskUtils
import numpy as np

def annToMask(mask_ann, h=None, w=None):
    try:
        if isinstance(mask_ann, list):
            rles = maskUtils.frPyObjects(mask_ann, h, w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, h, w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
    except Exception as e:
        raise ValueError(f"Invalid RLE for decode: {e}")
    return mask

def resize_nearest_like_torch(mask, out_h, out_w):
    H, W = mask.shape[-2:]

    scale_h = H / out_h
    scale_w = W / out_w

    ys = (np.arange(out_h) + 0.5) * scale_h - 0.5
    xs = (np.arange(out_w) + 0.5) * scale_w - 0.5

    ys = np.clip(np.round(ys), 0, H - 1).astype(np.int64)
    xs = np.clip(np.round(xs), 0, W - 1).astype(np.int64)

    return mask[..., ys[:, None], xs]

def iou_mask(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union > 0 else 0.0
