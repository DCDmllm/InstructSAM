import argparse
import sys
sys.path.append('./')
import re

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import os
import math
from tqdm import tqdm
from matplotlib import pyplot as plt
from pycocotools import mask as maskUtils
from transformers import AutoProcessor
from instructsam.models import load_pretrained_model
from instructsam import disable_torch_init, mm_infer_segmentation


# ---------------------------
# Original IoU helper (keep)
# ---------------------------
def compute_mask_IoU(masks, target):
    if target.sum() == 0 and masks.sum() == 0:
        return torch.tensor(0.0), torch.tensor(0.0), torch.tensor(1.0)
    temp = masks * target
    intersection = temp.sum(dim=-1)
    union = ((masks + target) - temp).sum(dim=-1)
    return intersection, union, intersection / (union + 1e-12)


# ---------------------------
# RLE / Mask utils
# ---------------------------
def _ensure_rle_counts_str(rle):
    if rle is None:
        return None
    rle = dict(rle)
    if isinstance(rle.get("counts", None), bytes):
        rle["counts"] = rle["counts"].decode("utf-8")
    return rle

def singleMask2rle(mask):
    if mask is None:
        return None
    rle = maskUtils.encode(np.array(mask[:, :, None], order='F', dtype="uint8"))[0]
    return _ensure_rle_counts_str(rle)

def annToMask(mask_ann, h=None, w=None):
    if isinstance(mask_ann, list):
        rles = maskUtils.frPyObjects(mask_ann, h, w)
        rle = maskUtils.merge(rles)
    elif isinstance(mask_ann['counts'], list):
        rle = maskUtils.frPyObjects(mask_ann, h, w)
    else:
        rle = mask_ann
    mask = maskUtils.decode(rle)
    return mask

def annToRLE_list(mask_ann, h, w):
    """
    Convert annotation to list of per-instance RLE dicts (0/1/N instances).
    Very defensive to handle common refcoco/grefcoco variants.
    """
    if mask_ann is None:
        return []

    # dict: RLE
    if isinstance(mask_ann, dict):
        if isinstance(mask_ann.get("counts", None), list):
            rle = maskUtils.frPyObjects(mask_ann, h, w)
            return [_ensure_rle_counts_str(rle)]
        return [_ensure_rle_counts_str(mask_ann)]

    # list: could be polygon / multi-instance
    if isinstance(mask_ann, list):
        if len(mask_ann) == 0:
            return []

        # polygon single instance: [[x1,y1,...], [x1,y1,...]]
        if isinstance(mask_ann[0], (list, tuple)) and len(mask_ann[0]) > 0 and isinstance(mask_ann[0][0], (int, float)):
            rles = maskUtils.frPyObjects(mask_ann, h, w)
            rle = maskUtils.merge(rles)
            return [_ensure_rle_counts_str(rle)]

        # list of dicts: multi-instance RLEs
        if isinstance(mask_ann[0], dict):
            out = []
            for inst in mask_ann:
                out.extend(annToRLE_list(inst, h, w))
            return out

        # list of polygon-per-instance: [ [ [..], ..], [ [..], ..], ... ]
        if isinstance(mask_ann[0], (list, tuple)) and len(mask_ann[0]) > 0 and isinstance(mask_ann[0][0], (list, tuple)):
            out = []
            for inst_poly in mask_ann:
                out.extend(annToRLE_list(inst_poly, h, w))
            return out

        # fallback
        try:
            m = annToMask(mask_ann, h, w)
            return [singleMask2rle(m.astype(np.uint8))]
        except Exception:
            return []

    return []


def union_mask_from_rles(rles, h, w):
    """Union multiple instance RLEs into a single HxW uint8 mask."""
    if not rles:
        return np.zeros((h, w), dtype=np.uint8)
    u = np.zeros((h, w), dtype=np.uint8)
    for r in rles:
        if r is None:
            continue
        m = maskUtils.decode(r).astype(np.uint8)
        # decode might return HxW or HxWx1
        if m.ndim == 3:
            m = m[:, :, 0]
        u = np.maximum(u, m)
    return u


# ---------------------------
# AP evaluation (Protocol A)
# ---------------------------
def _compute_ap_from_pr(rec, prec):
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

def evaluate_ap_from_records(records, iou_thresholds=None):
    """
    records: list of dict, each contains:
      - query_id (unique)
      - gt_rles: list of GT instance RLEs (0/1/N)
      - pred_rles: list of predicted instance RLEs
      - pred_scores: list of scores aligned with pred_rles
    """
    if iou_thresholds is None:
        iou_thresholds = [round(x, 2) for x in np.arange(0.50, 0.96, 0.05)]

    # GT map + global prediction list
    gts = {}
    preds = []  # (score, qid, pred_rle)
    total_gt = 0

    for r in records:
        qid = str(r["query_id"])
        gt_rles = r.get("gt_rles", []) or []
        pred_rles = r.get("pred_rles", []) or []
        pred_scores = r.get("pred_scores", []) or []

        gts[qid] = gt_rles
        total_gt += len(gt_rles)

        for prle, sc in zip(pred_rles, pred_scores):
            if prle is None:
                continue
            preds.append((float(sc), qid, prle))

    preds.sort(key=lambda x: x[0], reverse=True)

    out = {}
    aps = []

    for thr in iou_thresholds:
        matched = {qid: np.zeros(len(gt_list), dtype=bool) for qid, gt_list in gts.items()}
        tp = np.zeros(len(preds), dtype=np.float64)
        fp = np.zeros(len(preds), dtype=np.float64)

        for i, (score, qid, prle) in enumerate(preds):
            gt_list = gts.get(qid, [])
            if len(gt_list) == 0:
                fp[i] = 1.0
                continue

            used = matched[qid]
            iscrowd = np.zeros((len(gt_list),), dtype=np.uint8)
            ious = maskUtils.iou([prle], gt_list, iscrowd)[0]  # (len(gt_list),)

            best_iou = -1.0
            best_j = -1
            for j in range(len(gt_list)):
                if used[j]:
                    continue
                if float(ious[j]) > best_iou:
                    best_iou = float(ious[j])
                    best_j = j

            if best_iou >= thr and best_j >= 0:
                tp[i] = 1.0
                used[best_j] = True
            else:
                fp[i] = 1.0

        if total_gt == 0:
            ap = 0.0
        else:
            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            rec = tp_cum / float(total_gt)
            prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
            ap = _compute_ap_from_pr(rec, prec)

        out[f"AP@{thr:.2f}"] = ap
        aps.append(ap)

    out["mAP"] = float(np.mean(aps)) if aps else 0.0
    out["AP50"] = out.get("AP@0.50", 0.0)
    out["AP75"] = out.get("AP@0.75", 0.0)
    out["num_gt_instances"] = float(total_gt)
    out["num_pred_instances"] = float(len(preds))
    return out


# ---------------------------
# Chunk helpers
# ---------------------------
def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# ---------------------------
# Dataset
# ---------------------------
class Refcoco(Dataset):
    def __init__(self, image_folder, data_list, data_type=None, only_mask_img=True):
        data_list_new = []
        for d in data_list:
            image_path = os.path.join(image_folder, d['image'])
            for ann in d['annotation']:
                if 'Please output the segmentation mask.' not in ann['text']:
                    ann['text'] += ' Please output the segmentation mask.'
                if 'mask' in ann:
                    data_list_new.append(
                        {
                            "image": image_path,
                            "height": d["height"],
                            "width": d["width"],
                            "instruction": ann["text"],
                            "gt_mask": ann["mask"],
                        }
                    )
                else:
                    data_list_new.append(
                        {
                            "image": image_path,
                            "height": d["height"],
                            "width": d["width"],
                            "instruction": ann["text"],
                            "gt_mask": None,
                        }
                    )
        self.data_list = data_list_new

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        instruction = data['instruction']

        h = int(data["height"])
        w = int(data["width"])

        # For AP: keep per-instance GT rles (0/1/N)
        if data["gt_mask"] is None:
            gt_rles = []
            gt_union_mask = np.zeros((h, w), dtype=np.uint8)
        else:
            gt_rles = annToRLE_list(data["gt_mask"], h, w)
            gt_union_mask = union_mask_from_rles(gt_rles, h, w)

        contents = []
        contents.append({"type": "image", "image": data['image']})
        contents.append({"type": "text", "text": instruction})
        conversation = [{"role": "user", "content": contents}]

        return {
            "idx": idx,
            "conversation": conversation,
            "image_path": data["image"],
            "instruction": instruction,
            "height": h,
            "width": w,

            # both evaluations
            "gt_rles": gt_rles,
            "gt_union_mask": gt_union_mask,   # numpy uint8 HxW

            # debug
            "gt_mask_rle": data["gt_mask"],
        }

def collate_fn(batch):
    idx = [x['idx'] for x in batch]
    conv = [x['conversation'] for x in batch]
    ip = [x['image_path'] for x in batch]
    inst = [x['instruction'] for x in batch]
    h = [x['height'] for x in batch]
    w = [x['width'] for x in batch]
    gt_rles = [x['gt_rles'] for x in batch]
    gt_union = [x['gt_union_mask'] for x in batch]
    gt_raw = [x['gt_mask_rle'] for x in batch]
    return idx, conv, ip, inst, h, w, gt_rles, gt_union, gt_raw

def build_eval_dataloader(args, processor, distributed):
    questions = json.load(open(args.question_file))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    dataset = Refcoco(args.image_folder, questions)

    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        sampler = None
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        sampler=sampler
    )
    return dataloader


# ---------------------------
# Save results: keep giou/ciou + add AP
# ---------------------------
def _count_bin(n: int) -> str:
    if n == 0:
        return "0"
    elif n == 1:
        return "1"
    # elif 2 <= n <= 5:
    #     return "2-5"
    else:
        return ">=2"


def _compute_metrics_for_records(recs):
    """Compute giou/ciou + AP metrics for a list of record dicts."""
    if recs is None or len(recs) == 0:
        # keep keys stable even if empty
        ap_metrics = evaluate_ap_from_records(
            [],
            iou_thresholds=[round(x, 2) for x in np.arange(0.50, 0.96, 0.05)]
        )
        return {
            "giou": 0.0,
            "ciou": 0.0,
            **ap_metrics,
            "num_samples": 0,
        }

    giou = sum(d.get('iou', 0.0) for d in recs) / len(recs)
    inter_ = sum(d.get('inter', 0.0) for d in recs)
    union_ = sum(d.get('union', 0.0) for d in recs)
    ciou = inter_ / (union_ + 1e-10)

    ap_metrics = evaluate_ap_from_records(
        recs,
        iou_thresholds=[round(x, 2) for x in np.arange(0.50, 0.96, 0.05)]
    )

    return {
        "giou": float(giou),
        "ciou": float(ciou),
        **ap_metrics,
        "num_samples": int(len(recs)),
    }


def save_results(result, save_path):
    # -----------------------
    # Overall metrics
    # -----------------------
    overall_metrics = _compute_metrics_for_records(result)

    # -----------------------
    # Metrics by GT object count bins
    # -----------------------
    bins = {"0": [], "1": [],  ">=2": []}
    for r in result:
        n = int(r.get("gt_num_objects", len(r.get("gt_rles", []) or [])))
        bins[_count_bin(n)].append(r)

    by_num_objects = {}
    for k in ["0", "1", ">=2"]:
        by_num_objects[k] = _compute_metrics_for_records(bins[k])

    # pack final metrics
    metrics = {
        **overall_metrics,
        "by_num_objects": by_num_objects,
    }

    # prepend metrics entry as before
    result.insert(0, metrics)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if save_path.endswith(".json"):
        with open(save_path, "w") as f:
            json.dump(result, f, indent=4)
    elif save_path.endswith(".jsonl"):
        with open(save_path, "w") as f:
            for info in result:
                f.write(json.dumps(info) + "\n")
    else:
        raise ValueError("Unsupported file format.")
    print(f"Answer saved at:{save_path}")

# ---------------------------
# Inference loop
# ---------------------------
def run_inference(args):
    distributed = os.getenv('WORLD_SIZE', '1') > '1'
    if distributed:
        dist.init_process_group(backend="gloo")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        global_rank = dist.get_rank()
        device_map = {"": local_rank}

        disable_torch_init()
        tokenizer, model, processor = load_pretrained_model(
            args.model_path,
            None,
            attn_implementation='sdpa',
            device_map=device_map,
        )
        processor = AutoProcessor.from_pretrained(args.model_path)
    else:
        local_rank = 0
        global_rank = 0
        device_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else {"": "cpu"}

        disable_torch_init()
        tokenizer, model, processor = load_pretrained_model(
            args.model_path,
            None,
            attn_implementation='sdpa',
            device_map=device_map,
        )
        processor = AutoProcessor.from_pretrained(args.model_path)

    model.to(torch.bfloat16)

    val_loader = build_eval_dataloader(args, processor, distributed)

    results = []
    for i, (idx, conversation, image_paths, instruction, hs, ws, gt_rles_list, gt_union_list, gt_raw_list) in enumerate(
        tqdm(val_loader, desc=f"Rank {global_rank}", total=len(val_loader), position=local_rank)
    ):
        # keep batch_size=1 usage (consistent with your original code)
        idx = idx[0]
        conversation = conversation[0]
        image_path = image_paths[0]
        instruction = instruction[0]
        h = int(hs[0])
        w = int(ws[0])
        gt_rles = gt_rles_list[0] or []
        gt_union_mask = gt_union_list[0]  # numpy HxW uint8
        gt_mask_rle_raw = gt_raw_list[0]

        output, masks, cls_scores = mm_infer_segmentation(
            image_path,
            processor,
            conversation,
            model,
            tokenizer,
        )

        # -----------------------
        # Build predictions:
        # - For AP: instance rles + scores
        # - For giou/ciou: union pred mask
        # -----------------------
        pred_rles = []
        pred_scores = []

        if masks is not None and cls_scores is not None:
            keep = cls_scores > args.threshold
            selected_masks = masks[keep]
            selected_scores = cls_scores[keep]

            if selected_masks is not None and selected_masks.numel() > 0:
                selected_masks = F.interpolate(
                    selected_masks.unsqueeze(0),
                    size=(h, w),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0) > 0

                for pm, sc in zip(selected_masks, selected_scores):
                    rle = singleMask2rle(pm.detach().cpu().numpy().astype(np.uint8))
                    pred_rles.append(rle)
                    pred_scores.append(float(sc.detach().cpu().item()))

                pred_union = selected_masks.any(dim=0)  # HxW bool
            else:
                pred_union = torch.zeros((h, w), dtype=torch.bool, device=model.device)
        else:
            pred_union = torch.zeros((h, w), dtype=torch.bool, device=model.device)

        # -----------------------
        # giou/ciou: union IoU (same definition as your original)
        # -----------------------
        gt_union_t = torch.from_numpy(gt_union_mask).to(pred_union.device).bool()
        pred_masks = pred_union[None, None].float()  # 1x1xHxW
        gt_masks = gt_union_t[None, None].float()    # 1x1xHxW

        inter, union, iou = compute_mask_IoU(
            pred_masks.contiguous().view(1, -1),
            gt_masks.contiguous().view(1, -1)
        )

        record = {
            "idx": idx,
            "query_id": idx,  # for AP matching key
            "instruction": instruction,
            "prediction": output,
            "image_path": image_path,

            # giou/ciou components
            "inter": float(inter),
            "union": float(union),
            "iou": float(iou),

            # AP components
            "gt_rles": gt_rles,           # 0/1/N GT instances
            "pred_rles": pred_rles,       # 0/1/N pred instances
            "pred_scores": pred_scores,   # aligned scores

            "gt_num_objects": int(len(gt_rles)),

            # debug
            "gt_mask_rle": gt_mask_rle_raw,
        }
        results.append(record)

    if distributed:
        torch.cuda.empty_cache()
        gathered_results = [None for _ in range(dist.get_world_size())]
        dist.gather_object(
            obj=results,
            object_gather_list=gathered_results if global_rank == 0 else None,
            dst=0,
        )
        if global_rank == 0:
            results = sum(gathered_results, [])
            save_results(results, args.output_file)
        dist.destroy_process_group()
    else:
        save_results(results, args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='', required=True)
    parser.add_argument('--image_folder', help='Directory containing video files.', required=True)
    parser.add_argument('--question_file', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--output_file', help='Directory to save the model results JSON.', default='visualization/output.json')
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--vis", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.3)
    args = parser.parse_args()

    run_inference(args)
