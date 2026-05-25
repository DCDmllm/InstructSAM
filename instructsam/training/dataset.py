import os
import traceback
import random
from math import ceil
from typing import Optional, List, Dict, Any
import copy
import hashlib
import json
import pickle
import torch
import numpy as np
from PIL import Image, ImageOps
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import ProcessorMixin, logging, PretrainedConfig
from .utils import rank0_print, SEG_IMAGE_QUESTIONS_PHRASE, SEG_VIDEO_QUESTIONS_PHRASE, SEG_IMAGE_QUESTIONS_OCR, SEG_IMAGE_QUESTIONS_PHRASE_MULTI, clean_phrase
from .mm_utils import annToMask, resize_nearest_like_torch, iou_mask
from ..constants import SEG_TOKEN, REF_START_TOKEN, REF_END_TOKEN, SEG_START_TOKEN, SEG_END_TOKEN, IGNORE_INDEX, MAX_PHRASE_NUM, MAX_OBJ_NUM

logger = logging.get_logger(__name__)

def _get_rope_index_qwen3_vl(
    model_config: PretrainedConfig,
    input_ids: torch.LongTensor,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    """Different from the original implementation, Qwen3VL use timestamps rather than absolute time position ids."""

    # Since we use timestamps to seperate videos, like <t1> <vision_start> <frame1> <vision_end> <t2> <vision_start> <frame2> <vision_end>, the video_grid_thw should also be split
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    spatial_merge_size = model_config.vision_config.spatial_merge_size
    image_token_id = model_config.image_token_id
    video_token_id = model_config.video_token_id
    vision_start_token_id = model_config.vision_start_token_id
    mrope_position_deltas = []
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]
            image_nums, video_nums = 0, 0
            vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image

                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                # t_index is always 0 because llm_grid_t is always 1 (we use timestamps to encode the temporal information for videos)
                t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        return position_ids


def _get_rope_index(
    model_config: PretrainedConfig,
    input_ids: torch.LongTensor,
    **kwargs,
) -> torch.Tensor:
    if model_config.model_type == "qwen3_vl":
        position_ids = _get_rope_index_qwen3_vl(
            model_config=model_config,
            input_ids=input_ids,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported model: {model_config.model_type}")
    return position_ids

def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor

class DataCollator(object):
    def __init__(
        self,
        processor: ProcessorMixin,
        sequence_packing: bool,
    ):
        self.processor = processor
        self.sequence_packing = sequence_packing

    def _collate_mm_inputs(self, instances):
        mm_input_names = set(
            self.processor.image_processor.model_input_names + self.processor.video_processor.model_input_names
        )

        mm_inputs = {}
        for key in mm_input_names:
            data_list = [instance[key] for instance in instances if key in instance]
            if len(data_list) > 0:
                mm_inputs[key] = torch.cat(data_list, dim=0)

        return mm_inputs

    def _collate_fn_packing(self, instances):
        input_ids, position_ids, labels = [], [], []
        for instance in instances:
            input_ids.append(instance["input_ids"])
            if "position_ids" in instance:
                position_ids.append(instance["position_ids"])
            else:
                position_ids.append(torch.arange(instance["input_ids"].size(-1)).unsqueeze(0))
            tmp_labels = instance["labels"].clone()
            tmp_labels[..., 0] = -100
            labels.append(tmp_labels)

        batch = {
            "data_indices": [instance["data_index"] for instance in instances],
            "input_ids": torch.cat(input_ids, dim=-1),
            "position_ids": torch.cat(position_ids, dim=-1),
            "labels": torch.cat(labels, dim=-1),
            **self._collate_mm_inputs(instances),
        }

        batch["masks"] = [x["masks"] for x in instances]
        batch["sam_images"] = [x["sam_images"] for x in instances]
        batch["sam_size"] = [x["sam_size"] for x in instances]
        batch["masks_valid"] = [x["mask_valid"] for x in instances]
        batch["mask_type"] = [x["mask_type"] for x in instances]
        batch["mask_ids"] = [x["mask_ids"] for x in instances]

        return batch

    def _collate_fn_padding(self, instances):
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )
        input_ids = [ids.squeeze(0) for ids in input_ids]
        labels = [ids.squeeze(0) for ids in labels]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        position_ids = pad_and_cat(position_ids)

        batch = dict(
            data_indices=[instance["data_index"] for instance in instances],
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.processor.tokenizer.pad_token_id),
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw
        batch["position_ids"] = position_ids
        
        batch["phrase_ids"] = [x["phrase_ids"] for x in instances]
        batch["masks"] = [x["masks"] for x in instances]
        batch["sam_images"] = [x["sam_images"] for x in instances]
        batch["sam_size"] = [x["sam_size"] for x in instances]
        batch["masks_valid"] = [x["mask_valid"] for x in instances]
        batch["mask_type"] = [x["mask_type"] for x in instances]
        batch["mask_ids"] = [x["mask_ids"] for x in instances]

        return batch

    def __call__(self, instances: List[Dict[str, Any]]):
        if self.sequence_packing:
            return self._collate_fn_packing(instances)
        return self._collate_fn_padding(instances)


def preprocess_data_list(data_list, data_name="Dataset", max_seg_num=10):
    """
    Preprocess data list by splitting annotations according to mask/bbox/point limits.
    Print dataset size before and after processing, and per-sample split statistics.
    """

    processed_data = []

    rank0_print(f"[Preprocess] {data_name} Raw data size: {len(data_list)}")
    for idx, data in enumerate(data_list):
        original_len = len(processed_data)

        if "annotations" in data:
            data = copy.deepcopy(data)
            data["annotation"] = data.pop("annotations")

        if "annotation" not in data:
            processed_data.append(data)
            continue

        annotations = data["annotation"]
        random.shuffle(annotations)

        if "image" in data:
            MAX_MASK_NUM = 10
        elif "video" in data and len(data.get("frame_idx", [])) >= 32:
            MAX_MASK_NUM = 1
        elif "video" in data and len(data.get("frame_idx", [])) >= 16:
            MAX_MASK_NUM = 2
        else:
            MAX_MASK_NUM = 5

        mask_count = 0
        current_annotations = []

        for ann in annotations:
            if ann.get("mask") is None and ann.get("bbox") is None and ann.get("point") is None:
                current_annotations.append(ann)
                continue

            if "mask" in ann and len(ann["mask"]) > max_seg_num:
                continue
            if "bbox" in ann and len(ann["bbox"]) > max_seg_num:
                continue

            if "mask" in ann:
                mask_count += 1
            elif "bbox" in ann:
                mask_count += 1
            elif "point" in ann:
                mask_count += 1

            current_annotations.append(ann)

            if mask_count >= MAX_MASK_NUM:
                split_data = copy.deepcopy(data)
                split_data["annotation"] = current_annotations.copy()
                processed_data.append(split_data)

                current_annotations.clear()
                mask_count = 0

        if current_annotations:
            split_data = copy.deepcopy(data)
            split_data["annotation"] = current_annotations
            processed_data.append(split_data)

    rank0_print(f"[Preprocess] {data_name} Processed data size: {len(processed_data)}")

    return processed_data



class SFTDataset(Dataset):
    def __init__(
        self,
        model_config: PretrainedConfig,
        processor: ProcessorMixin,
        seg_processor,
        model_max_length: int,
        mm_max_length: int,
        fps: int,
        max_frames: int,
        dataloader_num_workers: Optional[int],
        data_args: str,
        requires_length: bool = False,
        mask_size: int = 288,
        use_multi_objs: bool = True,
    ):
        self.model_config = model_config
        self.processor = processor
        self.seg_processor = seg_processor
        self.model_max_length = model_max_length
        self.mm_max_length = mm_max_length
        self.fps = fps
        self.max_frames = max_frames
        self.data_root = data_args.data_root
        self.max_seg_nums = data_args.max_seg_nums
        self.mask_size = mask_size
        self.use_multi_objs = use_multi_objs
        self.skip_none = data_args.skip_none
        output_dir = data_args.output_dir

        self._dataset = self._load_data(data_args.ann_path, data_args.data_cache_dir, data_args)

    @property
    def modality_lengths(self):
        length_list = []
        for data_dict in self._dataset:
            mask_num = 0
            if "annotation" in data_dict and data_dict["annotation"] is not None:
                if isinstance(data_dict["annotation"], str):
                    data_dict["annotation"] = json.loads(data_dict["annotation"])
                for ann in data_dict["annotation"]:
                    if "ann" in ann and ann["ann"] is not None:
                        mask_num += 1
            elif "mask" in data_dict and data_dict["mask"] is not None:
                mask_num += len(data_dict['mask'])
            if mask_num == 0:
                mask_num = -1
            length_list.append(mask_num)

        return length_list

    def _load_data(self, data_path, cache_dir, data_args):
        def load_data_list(path):
            data_paths = []
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) == 1:
                        p, n = parts[0], 1
                    else:
                        p, n = parts[0], int(parts[1])
                    data_paths.extend([p] * n)
            return data_paths
        
        need_preprocess = True
        if '.txt' in data_path[0]:
            need_preprocess = True
            data_path = load_data_list(data_path[0])

        list_data_dict = []
        for d in data_path:
            rank0_print(f'begin load {d}...')
            dataset = load_dataset('json', data_files=os.path.join(data_args.data_path_root,d), cache_dir=cache_dir)['train']
            list_data_dict.append(dataset)
            # data_dict_inner = []
            # if d.endswith(".json"):
            #     data_dict_inner = json.load(open(os.path.join(data_args.data_path_root,d), "r"))
            # elif d.endswith(".jsonl"):
            #     with open(os.path.join(data_args.data_path_root,d), "r", encoding="utf-8") as fp:
            #         for line in fp:
            #             line = line.strip()
            #             obj = json.loads(line)
            #             data_dict_inner.append(obj)
            # else:
            #     raise Exception(f"Unsupported file format (<{d}>)!!!")
            
            # if need_preprocess:
            #     data_dict_inner = preprocess_data_list(data_dict_inner, d, self.max_seg_nums)
            # list_data_dict.extend(data_dict_inner)
        return concatenate_datasets(list_data_dict)

    def _convert_conversation(self, data_dict):
        data_folder = self.data_root

        mask_ids = []
        masks = []
        mask_type = 0 # 0: mask, 1: bbox, 2: point
        mask_valid = False
        new_conversation = []
        new_contents = []
        images = []
        phrase_str = ''

        if 'height' in data_dict:
            h = data_dict['height']
            w = data_dict['width']
        else:
            h = None
            w = None

        if 'image' in data_dict and data_dict['image'] is not None:
            modal = 'image'
            image_file = data_dict['image']
            image_file = os.path.join(self.data_root, image_file)
            new_contents.append({"type": "image", "image": image_file})
            images.append(image_file)
            # new_contents.append({"type": "image", "image": os.path.join(data_folder, image_file)})
            # images.append(os.path.join(data_folder, image_file))
                
            if "mask" in data_dict and data_dict["mask"] is not None and len(data_dict["mask"])>0: # mask
                for msk in data_dict["mask"]:
                    mask = annToMask(msk, h, w)
                    mask_ids.append([len(masks)])
                    masks.append(np.expand_dims(mask, axis=0))
                mask_valid = True

        elif 'video' in data_dict and data_dict['video'] is not None:
            modal = 'video'
            if isinstance(video_file, list):
                video = [os.path.join(self.data_root, frame) for frame in data_dict['video']]
                images+=video
                new_contents.append({"type": "video", "video": video})
            elif '.mp4' or '.avi' or '.mov' or '.mkv' or '.gif' in video_file:
                new_contents.append({"type": "video", "video": os.path.join(data_folder, video_file)})
            elif os.isdir(os.path.join(data_folder, data_dict['video'])):
                if 'frame_idx' not in data_dict:
                    raise ValueError("frame_idx is required for video")
                video_path = os.path.join(data_folder, data_dict['video'])
                frame_files = sorted(os.listdir(video_path))
                video = [os.path.join(video_path, frame_files[i]) for i in data_dict['frame_idx']]
                new_contents.append({"type": "video", "video": video})
                images+=video
            else:
                raise ValueError(f"Unsupported video format: {video_file}")

        else:
            modal = 'text'
        
        if "annotation" in data_dict and data_dict["annotation"] is not None: # grounding data
            if isinstance(data_dict["annotation"], str):
                data_dict["annotation"] = json.loads(data_dict["annotation"])
            conversation = []
            annotations = data_dict["annotation"]
            random.shuffle(annotations)
            phrase_num = 0
            obj_num = 0
            if self.use_multi_objs:
                if len(annotations)<=2:
                    multi_idx = -1
                elif modal=='image' and len(annotations)>0 and annotations[0]["type"]=='phrase':
                    multi_idx = len(annotations)-1
                    masks_multi_obj = []
                    while multi_idx>=0:
                        ann = annotations[multi_idx]
                        if ann["ann_type"]=="mask" and ann["ann"] is not None:
                            mask_all_multi_obj = np.zeros((1, h, w)) 
                            for msk in ann["ann"]:
                                msk = json.loads(msk)
                                mask_cur = np.expand_dims(annToMask(msk, h, w), axis=0) #[1,h,w]
                                masks_multi_obj.append(mask_cur)
                                mask_all_multi_obj = np.maximum(mask_all_multi_obj, mask_cur)
                                phrase_cur_multi_obj = annotations[multi_idx]["text"]
                            break
                        multi_idx -= 1
                else:
                    multi_idx = -1
            else:
                multi_idx = -1
            for i,annotation in enumerate(annotations):
                if i==multi_idx:
                    continue
                if self.skip_none and annotation["ann"] is None: # 不训none
                    continue
                ann_len = len(annotation["ann"]) if "ann" in annotation and annotation["ann"] is not None else 0
                if ann_len>self.max_seg_nums:
                    continue
                if obj_num+ann_len>MAX_OBJ_NUM or phrase_num+1>MAX_PHRASE_NUM:
                    break
                if annotation['type']=='phrase':
                    phrase = clean_phrase(annotation['text'])
                    if modal=='image':
                        new_contents.append({'type': 'text', 'text': random.choice(SEG_IMAGE_QUESTIONS_PHRASE).format(phrase=phrase)})
                    else:   
                        new_contents.append({'type': 'text', 'text': random.choice(SEG_VIDEO_QUESTIONS_PHRASE).format(phrase=phrase)})
                elif annotation['type']=='OCR':
                    phrase = annotation['text']
                    if modal=='image':
                        new_contents.append({'type': 'text', 'text': random.choice(SEG_IMAGE_QUESTIONS_OCR).format(phrase=phrase)})
                    else:
                        raise ValueError(f"No OCR question template found for video modality")
                elif annotation['type']=='sentence':
                    phrase = clean_phrase(annotation['category'])
                    new_contents.append({'type': 'text', 'text': annotation['text']})
                else:
                    raise ValueError(f"No phrase or sentence in the annotation")

                message = {"role": "user", "content": new_contents}
                new_conversation.append(message)
                new_contents = []

                mask_ids_inner = []
                if annotation["ann_type"]=="mask": #mask
                    if annotation["ann"] is None:
                        # new_contents.append({'type': 'text', 'text': f"{REF_START_TOKEN}{phrase}{REF_END_TOKEN}null"})
                        new_contents.append({'type': 'text', 'text': f"{REF_START_TOKEN}{phrase}{REF_END_TOKEN}{SEG_START_TOKEN}{SEG_TOKEN * self.max_seg_nums}{SEG_END_TOKEN}"})
                        phrase_str += f"{REF_START_TOKEN}{phrase}"
                        phrase_num +=1
                        message = {"role": "assistant", "content": new_contents}
                        new_conversation.append(message)
                        new_contents = []
                        if modal=='image':
                            mask_cur = np.zeros((1, h, w)) #[1,h,w]
                            mask_ids.append([len(masks)])
                            masks.append(mask_cur)
                        else:
                            raise ValueError(f'Null mask not supported for video modality yet!')
                        continue
                    else:
                        mask_valid = True
                        new_contents.append({'type': 'text', 'text': f"{REF_START_TOKEN}{phrase}{REF_END_TOKEN}{SEG_START_TOKEN}{SEG_TOKEN * self.max_seg_nums}{SEG_END_TOKEN}"})
                        phrase_str += f"{REF_START_TOKEN}{phrase}"
                        phrase_num +=1
                        obj_num += ann_len

                    # if 'segmentation_file' in data_dict:
                    #     mask_json = json.load(open(os.path.join(data_folder, data_dict['segmentation_file'])))
                    mask_all = np.zeros((1, h, w))
                    for msk in annotation["ann"]:
                        msk = json.loads(msk)
                        if modal=='image':
                            mask_cur = np.expand_dims(annToMask(msk, h, w), axis=0) #[1,h,w]
                            mask_all = np.maximum(mask_all, mask_cur)
                        else:
                            masks_video = []
                            for k in msk.keys():
                                if msk[k] is None:
                                    if 'height' in data_dict:
                                        mask = np.zeros((data_dict['height'], data_dict['width'])) 
                                    else:
                                        for k1 in msk.keys():
                                            if msk[k1] is not None:
                                                h, w = msk[k1]["size"]
                                                break
                                        mask = np.zeros((h, w)) 
                                else:
                                    mask = annToMask(msk[k], h, w)
                                masks_video.append(mask)
                            mask_cur = np.array(masks_video)

                        mask_ids_inner.append(len(masks))
                        masks.append(mask_cur)
                    mask_ids.append(mask_ids_inner)
                    # add multi phrase format
                    if multi_idx!=-1 and iou_mask(mask_all, mask_all_multi_obj)<0.05:
                        mask_ids_inner = []
                        for idx_ in range(len(masks_multi_obj)):
                            mask_ids_inner.append(len(masks)+idx_)
                        masks += masks_multi_obj
                        mask_ids.append(mask_ids_inner)
                        new_contents[-1]['text'] += f", {REF_START_TOKEN}{phrase_cur_multi_obj}{REF_END_TOKEN}{SEG_START_TOKEN}{SEG_TOKEN * self.max_seg_nums}{SEG_END_TOKEN}"
                        phrase_str += f"{REF_START_TOKEN}{phrase_cur_multi_obj}"
                        phrase_num += 1
                        obj_num += ann_len
                        new_conversation[-1]['content'][-1]['text'] = random.choice(SEG_IMAGE_QUESTIONS_PHRASE_MULTI).format(phrase1=phrase, phrase2=phrase_cur_multi_obj)
                        multi_idx = -1

                elif annotation["ann_type"]=="point":
                    mask_type = 2
                    pass
                elif annotation["ann_type"]=="bbox":
                    mask_type = 1
                    new_contents.append({'type': 'text', 'text': f"{REF_START_TOKEN}{phrase}{REF_END_TOKEN}{SEG_START_TOKEN}{SEG_TOKEN * self.max_seg_nums}{SEG_END_TOKEN}"})
                    phrase_str += f"{REF_START_TOKEN}{phrase}"
                    phrase_num +=1
                    obj_num += ann_len
                    for bbox in annotation["ann"]:
                        bbox = json.loads(bbox)
                        if modal=='image':
                            mask = np.zeros((h, w))
                            x1, y1, wb, hb = bbox
                            x1 = int(max(x1,0))
                            y1 = int(max(y1,0))
                            wb = int(min(wb, w - x1))
                            hb = int(min(hb, h - y1))
                            mask[y1:y1+hb, x1:x1+wb] = 1
                            mask = np.expand_dims(mask, axis=0) #[1,h,w]
                        else:
                            raise ValueError(f'Bounding box not supported for video modality yet!')
                        mask_ids_inner.append(len(masks))
                        masks.append(mask)
                    mask_ids.append(mask_ids_inner)

                message = {"role": "assistant", "content": new_contents}
                new_conversation.append(message)
                new_contents = []
                # if obj_num>=MAX_OBJ_NUM or phrase_num>=MAX_PHRASE_NUM:
                #     break
        if len(masks)>0:
            masks = np.array(masks)
            # masks = resize_nearest_like_torch(masks, self.mask_size, self.mask_size)
            masks = torch.from_numpy(masks)
        else:
            masks = None

        if 'conversations' in data_dict and data_dict['conversations'] is not None:
            if isinstance(data_dict['conversations'], str):
                data_dict['conversations'] = json.loads(data_dict['conversations'])
            for idx, conv in enumerate(data_dict['conversations']):
                new_contents.append({'type': 'text', 'text': conv['value'].replace('<image>','').replace('<video>','').strip()})
                if idx%2==0:
                    message = {"role": "user", "content": new_contents}
                else:
                    message = {"role": "assistant", "content": new_contents}
                new_conversation.append(message)
                new_contents = []
                # phrase_str+=conv['value']

        sam_images = []
        if images is not None and len(images)>0:
            sam_size = None
            for image in images:
                image = Image.open(image)
                image = ImageOps.exif_transpose(image).convert("RGB")
                sam_inputs = self.seg_processor(image)
                sam_images.append(sam_inputs['pixel_values'][0])
                sam_size = sam_inputs.original_sizes[0]
            sam_images = torch.cat(sam_images, dim=0)
        if len(sam_images)==0:
            sam_size = (1008, 1008)
            sam_images = torch.zeros(3, sam_size[0], sam_size[1])

        # if not mask_valid:
        # print(new_conversation)
        # print('**************')
        # print(phrase_str)
        # print('==============')
        # if len(new_conversation)==0:
        #     print(data_dict)
        #     print('==============')

        return new_conversation, sam_images, sam_size, masks, mask_ids, mask_valid, mask_type, phrase_str

    def _convert_conversation_instseg(self, data_dict):
        data_folder = self.data_root

        mask_ids = []
        masks = []
        mask_type = 0 # 0: mask, 1: bbox, 2: point
        mask_valid = False
        new_conversation = []
        new_contents = []
        images = []
        phrase_str = ''

        if 'height' in data_dict:
            h = data_dict['height']
            w = data_dict['width']
        else:
            h = None
            w = None

        if 'image' in data_dict and data_dict['image'] is not None:
            modal = 'image'
            image_file = data_dict['image']
            image_file = os.path.join(self.data_root, image_file)
            new_contents.append({"type": "image", "image": image_file})
            images.append(image_file)

        elif 'video' in data_dict and data_dict['video'] is not None:
            modal = 'video'
            if isinstance(video_file, list):
                video = [os.path.join(self.data_root, frame) for frame in data_dict['video']]
                images+=video
                new_contents.append({"type": "video", "video": video})
            elif isinstance(video_file, str) and video_file.lower().endswith(('.mp4','.avi','.mov','.mkv','.gif')):
                new_contents.append({"type": "video", "video": os.path.join(data_folder, video_file)})
            elif os.path.isdir(os.path.join(data_folder, data_dict['video'])):
                if 'frame_idx' not in data_dict:
                    raise ValueError("frame_idx is required for video")
                video_path = os.path.join(data_folder, data_dict['video'])
                frame_files = sorted(os.listdir(video_path))
                video = [os.path.join(video_path, frame_files[i]) for i in data_dict['frame_idx']]
                new_contents.append({"type": "video", "video": video})
                images+=video
            else:
                raise ValueError(f"Unsupported video format: {video_file}")

        else:
            modal = 'text'
        
        if "annotation" in data_dict and data_dict["annotation"] is not None: # grounding data
            if isinstance(data_dict["annotation"], str):
                data_dict["annotation"] = json.loads(data_dict["annotation"])
            conversation = []
            annotations = data_dict["annotation"]
            random.shuffle(annotations)
            phrase_num = 0
            obj_num = 0
            phrase_list = []
            answer_str = ''
            for i,annotation in enumerate(annotations):
                ann_len = len(annotation["ann"]) if "ann" in annotation and annotation["ann"] is not None else 0
                if ann_len>self.max_seg_nums:
                    continue
                if obj_num+ann_len>MAX_OBJ_NUM or phrase_num+1>MAX_PHRASE_NUM:
                    break
                if self.skip_none and annotation["ann"] is None: # 不训none
                    continue
                if annotation['type']=='phrase':
                    phrase = clean_phrase(annotation['text'])
                    phrase_list.append(phrase)
                    
                else:
                    raise ValueError(f"No phrase or sentence in the annotation")


                mask_ids_inner = []
                if annotation["ann_type"]=="mask": #mask
                    if annotation["ann"] is None:
                        # answer_str += f"{REF_START_TOKEN}{phrase}{REF_END_TOKEN}null, "
                        answer_str += f"{REF_START_TOKEN}{phrase}{REF_END_TOKEN}{SEG_START_TOKEN}{SEG_TOKEN * self.max_seg_nums}{SEG_END_TOKEN}, "
                        phrase_str += f"{REF_START_TOKEN}{phrase}"
                        phrase_num +=1
                        if modal=='image':
                            mask_cur = np.zeros((1, h, w)) #[1,h,w]
                            mask_ids.append([len(masks)])
                            masks.append(mask_cur)
                        continue
                    else:
                        mask_valid = True
                        answer_str += f"{REF_START_TOKEN}{phrase}{REF_END_TOKEN}{SEG_START_TOKEN}{SEG_TOKEN * self.max_seg_nums}{SEG_END_TOKEN}, "
                        phrase_str += f"{REF_START_TOKEN}{phrase}"
                        phrase_num +=1
                        obj_num += ann_len

                    # if 'segmentation_file' in data_dict:
                    #     mask_json = json.load(open(os.path.join(data_folder, data_dict['segmentation_file'])))
                    for msk in annotation["ann"]:
                        msk = json.loads(msk)
                        if modal=='image':
                            mask_cur = np.expand_dims(annToMask(msk, h, w), axis=0) #[1,h,w]
                        else:
                            masks_video = []
                            for k in msk.keys():
                                if msk[k] is None:
                                    if 'height' in data_dict:
                                        mask = np.zeros((data_dict['height'], data_dict['width'])) 
                                    else:
                                        for k1 in msk.keys():
                                            if msk[k1] is not None:
                                                h, w = msk[k1]["size"]
                                                break
                                        mask = np.zeros((h, w)) 
                                else:
                                    mask = annToMask(msk[k], h, w)
                                masks_video.append(mask)
                            mask_cur = np.array(masks_video)

                        mask_ids_inner.append(len(masks))
                        masks.append(mask_cur)
                    mask_ids.append(mask_ids_inner)
                        
                elif annotation["ann_type"]=="point":
                    mask_type = 2
                    pass
                elif annotation["ann_type"]=="bbox":
                    mask_type = 1
                    answer_str += f"{REF_START_TOKEN}{phrase}{REF_END_TOKEN}{SEG_START_TOKEN}{SEG_TOKEN * self.max_seg_nums}{SEG_END_TOKEN}, "
                    phrase_str += f"{REF_START_TOKEN}{phrase}"
                    phrase_num +=1
                    obj_num += ann_len
                    for bbox in annotation["ann"]:
                        bbox = json.loads(bbox)
                        if modal=='image':
                            mask = np.zeros((h, w))
                            x1, y1, wb, hb = bbox
                            x1 = int(max(x1,0))
                            y1 = int(max(y1,0))
                            wb = int(min(wb, w - x1))
                            hb = int(min(hb, h - y1))
                            mask[y1:y1+hb, x1:x1+wb] = 1
                            mask = np.expand_dims(mask, axis=0) #[1,h,w]
                        else:
                            raise ValueError(f'Bounding box not supported for video modality yet!')
                        mask_ids_inner.append(len(masks))
                        masks.append(mask)
                    mask_ids.append(mask_ids_inner)

                # if obj_num>=MAX_OBJ_NUM or phrase_num>=MAX_PHRASE_NUM:
                #     break
        
        phrase_str_all = ', '.join(phrase_list)
        # 最后一个换成 and
        if len(phrase_list)>=2:
            last_comma_idx = phrase_str_all.rfind(',')
            phrase_str_all = phrase_str_all[:last_comma_idx] + ' and' + phrase_str_all[last_comma_idx+1:]
        if modal=='image':
            new_contents.append({'type': 'text', 'text': random.choice(SEG_IMAGE_QUESTIONS_PHRASE).format(phrase=phrase_str_all)})
        else:   
            new_contents.append({'type': 'text', 'text': random.choice(SEG_VIDEO_QUESTIONS_PHRASE).format(phrase=phrase_str_all)})
        
        new_conversation.append({"role": "user", "content": new_contents})
        new_conversation.append({"role": "assistant", "content": [
            {'type': 'text', 'text': answer_str.strip().rstrip(',').strip()} # +'.'
        ]})
        # print(new_conversation)

        if len(masks)>0:
            masks = np.array(masks)
            # masks = resize_nearest_like_torch(masks, self.mask_size, self.mask_size)
            masks = torch.from_numpy(masks)
        else:
            masks = None

        if 'conversations' in data_dict:
            for idx, conv in enumerate(data_dict['conversations']):
                new_contents.append({'type': 'text', 'text': conv['value']})
                if idx%2==0:
                    message = {"role": "user", "content": new_contents}
                else:
                    message = {"role": "assistant", "content": new_contents}
                new_conversation.append(message)
                new_contents = []
                phrase_str+=conv['value']

        sam_images = []
        if images is not None and len(images)>0:
            sam_size = None
            for image in images:
                image = Image.open(image)
                image = ImageOps.exif_transpose(image).convert("RGB")
                sam_inputs = self.seg_processor(image)
                sam_images.append(sam_inputs['pixel_values'][0])
                sam_size = sam_inputs.original_sizes[0]
            sam_images = torch.cat(sam_images, dim=0)
        
        if len(sam_images)==0:
            sam_size = (1008, 1008)
            sam_images = torch.zeros(3, sam_size[0], sam_size[1])

        # if not mask_valid:
        # print(new_conversation)
        # print('**************')
        # print(phrase_str)
        # print('==============')

        return new_conversation, sam_images, sam_size, masks, mask_ids, mask_valid, mask_type, phrase_str


    def _preprocess(self, data_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        if data_dict["type"]=="instseg":
            if random.random()>0.5:
                conversation, sam_images, sam_size, masks, mask_ids, mask_valid, mask_type, phrase_str = self._convert_conversation_instseg(data_dict)
            else:
                conversation, sam_images, sam_size, masks, mask_ids, mask_valid, mask_type, phrase_str = self._convert_conversation(data_dict)
        else:
            conversation, sam_images, sam_size, masks, mask_ids, mask_valid, mask_type, phrase_str = self._convert_conversation(data_dict)

        model_inputs = self.processor.apply_chat_template(
            conversation=conversation,
            mm_max_length=self.mm_max_length,
            fps=self.fps,
            max_frames=self.max_frames,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            return_labels=True,
        )
        phrase_ids = self.processor.tokenizer.encode(phrase_str)
        model_inputs["phrase_ids"] = phrase_ids
        model_inputs["sam_images"] = sam_images
        model_inputs["sam_size"] = sam_size
        model_inputs["masks"] = masks
        model_inputs["mask_ids"] = mask_ids
        model_inputs["mask_valid"] = mask_valid
        model_inputs["mask_type"] = mask_type

        # print(self.processor.decode(model_inputs["input_ids"][0], skip_special_tokens=True))
        # for token_id, label in zip(model_inputs["input_ids"][0], model_inputs["labels"][0]):
        #     token = self.processor.decode([token_id])
        #     if token == "<|image_pad|>":
        #         continue
        #     print([token_id, token, label])
        # exit()

        assert model_inputs["input_ids"].size(-1) <= self.model_max_length, (
            f"Sequence length ({model_inputs['input_ids'].size(-1)}) exceeds model max length ({self.model_max_length})"
        )

        model_inputs["position_ids"] = _get_rope_index(
            model_config=self.model_config,
            **model_inputs,
        )

        return model_inputs

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        if self._dataset[index]['image'] is not None and 'objects365_v2_01808559.jpg' in self._dataset[index]['image']:
            backup_idx = random.randint(0, len(self) - 1)
            print(f"Encounted error when process {index}-th example, use {backup_idx}-th example instead!!!")
            return self.__getitem__(backup_idx)
        try:
            # print('begin process {}-th example... image: {}'.format(index, self._dataset[index]['image']))
            data_dict = self._preprocess(self._dataset[index])
            data_dict["data_index"] = index
            # print('end process {}-th example.'.format(index))
        except Exception:
            traceback.print_exc()
            backup_idx = random.randint(0, len(self) - 1)
            print(f"Encounted error when process {index}-th example, use {backup_idx}-th example instead!!!")
            return self.__getitem__(backup_idx)
        return data_dict

    def __len__(self):
        return len(self._dataset)

    def __repr__(self):
        return self._dataset.__repr__()
