import argparse
import sys
sys.path.append('./')
import re

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor
from instructsam.models import load_pretrained_model
from instructsam import disable_torch_init, mm_infer_segmentation
import json
import numpy as np
import os
import math
from tqdm import tqdm
from torchvision.transforms import v2
from matplotlib import pyplot as plt
from pycocotools import mask as maskUtils

def compute_mask_IoU(masks, target):
    temp = masks * target
    intersection = temp.sum()
    union = ((masks + target) - temp).sum()
    return intersection, union, (intersection / (union + 1e-12))

def singleMask2rle(mask):
    if mask is None:
        return None
    rle = maskUtils.encode(np.array(mask[:, :, None], order='F', dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle
    
def annToMask(mask_ann, h=None, w=None):
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
    return mask

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

class RoboRefIt(Dataset):
    def __init__(self, image_folder, data_list, data_type=None, only_mask_img=True):
        """
        RoboRefIt Dataset for referring expression segmentation
        
        Expected JSON format:
        [
            {
                "image": "relative/path/to/image.jpg",
                "question": "referring expression text",
                "masks": {"size": [H, W], "counts": "..."}  # RLE format
            },
            ...
        ]
        """
        self.data_list = []
        for d in data_list:
            self.data_list.append({
                "image": os.path.join(image_folder, d['image']),
                "question": d.get('question', d.get('expression', '')) + ' Please output the segmentation mask.',
                "gt_mask": d["masks"]
            })
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data = self.data_list[idx]
        
        instruction = data['question']
        
        # Load GT mask
        gt_mask = annToMask(data["gt_mask"])
        
        contents = []
        contents.append({"type": "image", "image": data["image"]})
        contents.append({"type": "text", "text": instruction})
        conversation = [{"role": "user", "content": contents}]
   
        return {
            'idx': idx,
            'image_path': data["image"],
            'masks': gt_mask,
            'instruction': instruction,
            'conversation': conversation,
            'gt_mask_rle': data["gt_mask"]
        }

def collate_fn(batch):
    idx = [x['idx'] for x in batch]
    image_path = [x['image_path'] for x in batch]
    masks = [x['masks'] for x in batch]
    instruction = [x['instruction'] for x in batch]
    conversation = [x['conversation'] for x in batch]
    gt_mask_rle = [x['gt_mask_rle'] for x in batch]
    
    return idx, image_path, masks, instruction, conversation, gt_mask_rle

def eval_model(args):
    # Check if distributed
    distributed = os.getenv('WORLD_SIZE', '1') > '1'
    if distributed:
        dist.init_process_group(backend="gloo")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()

        disable_torch_init()
        tokenizer, model, processor = load_pretrained_model(
            args.model_path,
            None,
            attn_implementation='sdpa',
            device_map={"": local_rank},
        )
        processor = AutoProcessor.from_pretrained(args.model_path)
    else:
        local_rank = 0
        global_rank = 0
        disable_torch_init()
        device_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else {"": "cpu"}
        tokenizer, model, processor = load_pretrained_model(
            args.model_path,
            None,
            attn_implementation='sdpa',
            device_map=device_map,
        )
        processor = AutoProcessor.from_pretrained(args.model_path)

    model.to(torch.bfloat16)

    # Load questions
    questions = json.load(open(args.question_file, "r"))

    # Split data for distributed training
    if distributed:
        questions = questions[global_rank::world_size]
    else:
        questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    
    # Create dataset
    dataset = RoboRefIt(args.image_folder, questions)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, 
                           num_workers=args.num_workers, collate_fn=collate_fn)
    
    # Evaluation
    results = []
    
    for idx, image_path, masks, instruction, conversation, gt_mask_rle in tqdm(dataloader, desc="Evaluating"):
        try:
            idx = idx[0]
            gt_masks = masks[0]
            instruction = instruction[0]
            image_path = image_path[0]
            conversation = conversation[0]
            gt_mask_rle = gt_mask_rle[0]

            # Model inference
            output, masks, cls_scores = mm_infer_segmentation(
                image_path,
                processor,
                conversation,
                model=model,
                tokenizer=tokenizer,
            )

            h, w = gt_masks.shape[0], gt_masks.shape[1]
        
            if masks is not None and cls_scores is not None:
                keep = cls_scores > args.threshold
                selected_masks = masks[keep]

            if masks is None or cls_scores is None or selected_masks.numel() == 0:
                print(output)
                selected_masks = None
                pred_masks = torch.zeros((1, 1, h, w), dtype=bool, device=model.device)
            else:
                selected_masks = F.interpolate(selected_masks.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False).squeeze(0)>0
                selected_masks_ = selected_masks.any(dim=0)          
                pred_masks = selected_masks_[None].float().unsqueeze(0)  
                
            mask_rle = []
            if selected_masks is None:
                mask_rle.append(None)
            else:
                for pred_msk in selected_masks:
                    mask_rle.append(singleMask2rle(pred_msk.detach().cpu().numpy()))
            gt_masks = torch.from_numpy(gt_masks)
            
            intersection, union, iou = compute_mask_IoU(pred_masks, gt_masks)

            # Save result
            result = {
                'idx': idx,
                'instruction': instruction,
                'prediction': output,
                'image_path': image_path,
                'iou': float(iou.item()),
                'intersection': int(intersection.item()),
                'union': int(union.item()),
                'pred_mask_rle': mask_rle,
                'gt_mask_rle': gt_mask_rle
            }
            results.append(result)
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            results.append({
                'idx': idx,
                'instruction': instruction[0] if isinstance(instruction, list) else instruction,
                'image_path': image_path[0] if isinstance(image_path, list) else image_path,
                'iou': 0.0,
                'error': str(e)
            })
            continue
    
    return results

def main(args):
    # Check if distributed
    distributed = os.getenv('WORLD_SIZE', '1') > '1'

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    results = eval_model(args)

    if distributed:
        # Distributed evaluation
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Save rank results
        rank_file = args.output_file.replace('.json', f'_rank{global_rank}.json')
        with open(rank_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Wait for all ranks
        dist.barrier()

        # Merge results (only rank 0)
        if global_rank == 0:
            all_results = []
            for i in range(world_size):
                rank_file = args.output_file.replace('.json', f'_rank{i}.json')
                with open(rank_file, 'r') as f:
                    all_results.extend(json.load(f))
            
            # Calculate overall metrics
            ious = [r['iou'] for r in all_results if 'iou' in r]
            mean_iou = np.mean(ious) if ious else 0.0
            
            summary = {
                'mean_iou': float(mean_iou),
                'num_samples': len(all_results),
                'num_valid': len(ious)
            }
            
            # Save final results
            final_results = [summary] + all_results
            with open(args.output_file, 'w') as f:
                json.dump(final_results, f, indent=2)
            
            print(f"\n{'='*50}")
            print(f"RoboRefIt Evaluation Results")
            print(f"{'='*50}")
            print(f"Mean IoU: {mean_iou:.4f}")
            print(f"Total samples: {len(all_results)}")
            print(f"Valid samples: {len(ious)}")
            print(f"Results saved to: {args.output_file}")
            print(f"{'='*50}\n")
            
            # Clean up rank files
            for i in range(world_size):
                rank_file = args.output_file.replace('.json', f'_rank{i}.json')
                if os.path.exists(rank_file):
                    os.remove(rank_file)
    elif args.num_chunks > 1:
        # Single process evaluation
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        
        results = eval_model(args)
        
        # Calculate metrics
        ious = [r['iou'] for r in results if 'iou' in r]
        mean_iou = np.mean(ious) if ious else 0.0
        
        summary = {
            'mean_iou': float(mean_iou),
            'num_samples': len(results),
            'num_valid': len(ious)
        }
        
        # Save results
        final_results = [summary] + results
        with open(args.output_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"\n{'='*50}")
        print(f"RoboRefIt Evaluation Results")
        print(f"{'='*50}")
        print(f"Mean IoU: {mean_iou:.4f}")
        print(f"Total samples: {len(results)}")
        print(f"Valid samples: {len(ious)}")
        print(f"Results saved to: {args.output_file}")
        print(f"{'='*50}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='Path to model checkpoint', required=True)
    parser.add_argument('--image_folder', help='Directory containing image files', required=True)
    parser.add_argument('--question_file', help='Path to the question JSON file', required=True)
    parser.add_argument('--output_file', help='Path to save results JSON', required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    
    main(args)
