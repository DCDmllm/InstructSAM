import os
from typing import List, Dict, Any, Optional, Union, Callable
import copy
import ffmpeg
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
from transformers.image_utils import load_image
from transformers.video_utils import VideoMetadata

from ..training.utils import get_args, get_encoder_load_balancing_group
from torch import Tensor
import logging
from huggingface_hub import hf_hub_download
import functools

def select_vision_outputs(vout_batch, idx: int):
    vout = copy.copy(vout_batch) 
    vout.last_hidden_state = vout_batch.last_hidden_state[idx:idx+1]
    vout.fpn_hidden_states = tuple(
        feat[idx:idx+1] for feat in vout_batch.fpn_hidden_states
    )
    vout.fpn_position_encoding = tuple(
        feat[idx:idx+1] for feat in vout_batch.fpn_position_encoding
    )
    return vout

def expand_vision_features(vision_outputs, obj_num):
    new_outputs = copy.copy(vision_outputs)

    new_outputs['last_hidden_state'] = vision_outputs['last_hidden_state'].expand(obj_num, -1, -1)

    for k in ['fpn_hidden_states', 'fpn_position_encoding']:
        new_outputs[k] = tuple(
            feat.expand(obj_num, -1, -1, -1) for feat in vision_outputs[k]
        )

    return new_outputs

def get_phrase_embedding(input_ids, hidden_states, start_token_id, max_len=32):
    assert input_ids.dim() == 1, "input_ids should be [n]"
    assert hidden_states.dim() == 2 and hidden_states.size(0) == input_ids.size(0), \
        "hidden_states should be [n, hidden_dim] and align with input_ids"

    device = input_ids.device
    dtype = hidden_states.dtype
    hidden_dim = hidden_states.size(1)

    # 找到所有 start_token_id 的位置
    start_pos = (input_ids == start_token_id).nonzero(as_tuple=False).flatten()
    m = int(start_pos.numel())

    if m == 0:
        phrase_embeds = hidden_states.new_zeros((0, max_len, hidden_dim))
        attn_mask = torch.zeros((0, max_len), device=device, dtype=torch.bool)
        return phrase_embeds, attn_mask

    phrase_embeds = hidden_states.new_zeros((m, max_len, hidden_dim))
    attn_mask = torch.zeros((m, max_len), device=device, dtype=torch.bool)

    n = input_ids.size(0)
    for i, s in enumerate(start_pos.tolist()):
        seg_start = s + 1
        seg_end = start_pos[i + 1].item() if i + 1 < m else n  # 到下一个 start 或结尾

        if seg_start >= seg_end:
            continue  # 空 phrase

        seg_len = seg_end - seg_start
        take = min(seg_len, max_len)

        phrase_embeds[i, :take] = hidden_states[seg_start:seg_start + take]
        attn_mask[i, :take] = True
    return phrase_embeds, attn_mask

import torch

def get_phrase_ids_by_start_end(
    input_ids,
    START_TOKEN_ID=151646,
    END_TOKEN_ID=151647,
    max_len=32,
):
    device = input_ids.device
    assert input_ids.dim() == 2, "input_ids should be [B, T]"
    B, T = input_ids.shape

    phrases = []
    for b in range(B):
        ids = input_ids[b]
        t = 0
        while t < T:
            if ids[t] == START_TOKEN_ID:
                s = t
                t += 1
                while t < T and ids[t] != END_TOKEN_ID:
                    t += 1
                if t < T:  
                    mid = ids[s + 1:t]
                    phrases.append(mid)
            t += 1

    n = len(phrases)
    phrase_ids = torch.zeros((n, max_len), dtype=input_ids.dtype, device=device)
    attn_mask = torch.zeros((n, max_len), dtype=torch.bool, device=device)

    for i, mid in enumerate(phrases):
        full = torch.cat([
            torch.tensor([START_TOKEN_ID], device=device, dtype=input_ids.dtype),
            mid,
            torch.tensor([END_TOKEN_ID], device=device, dtype=input_ids.dtype),
        ])

        L = min(full.numel(), max_len)
        phrase_ids[i, :L] = full[:L]
        attn_mask[i, :L] = True

    return phrase_ids, attn_mask



def read_video_ffmpeg(
    video: str,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    fps: Optional[float] = None,
    max_frames: Optional[float] = None,
    precise_time: bool = False,
    verbose: bool = False,
):
    probe = ffmpeg.probe(video)
    duration = float(probe["format"]["duration"])
    video_stream = next((stream for stream in probe["streams"] if stream["codec_type"] == "video"), None)
    w, h = int(video_stream["width"]), int(video_stream["height"])
    video_fps = video_stream["avg_frame_rate"]
    if "/" in video_fps:
        numerator, denominator = map(int, video_fps.split("/"))
        if denominator == 0:
            video_fps = 0.0
        else:
            video_fps = numerator / denominator
    else:
        video_fps = float(video_fps)
    total_num_frames = round(video_fps * duration)

    kwargs, input_kwargs, output_kwargs = {}, {}, {}
    do_trim = start_time is not None or end_time is not None
    if start_time is not None:
        new_start_time = max(float(video_stream["start_time"]), start_time)
        duration -= new_start_time - start_time
        start_time = new_start_time
    else:
        start_time = float(video_stream["start_time"])
    if end_time is not None:
        duration = min(duration, end_time - start_time)
    else:
        duration = duration
    if do_trim:
        kwargs = {"ss": start_time, "t": duration}
    if precise_time:
        output_kwargs.update(kwargs)
    else:
        input_kwargs.update(kwargs)

    stream = ffmpeg.input(video, **input_kwargs)
    if fps is not None:
        stream = ffmpeg.filter(stream, "fps", fps=fps, round="near")
    stream = ffmpeg.output(stream, "pipe:", format="rawvideo", pix_fmt="rgb24", **output_kwargs)
    out, _ = ffmpeg.run(stream, capture_stdout=True, quiet=not verbose)

    frames = np.frombuffer(out, np.uint8).reshape([-1, h, w, 3]).transpose([0, 3, 1, 2]).copy()

    if fps is not None:
        timestamps = np.arange(start_time, start_time + duration + 1 / fps, 1 / fps)[: len(frames)]
        frames_indices = np.round(timestamps * video_fps)
    else:
        total_num_frames = len(frames)
        frames_indices = np.arange(total_num_frames)

    if max_frames is not None and len(frames) > max_frames:
        indices = np.round(np.linspace(0, len(frames) - 1, max_frames)).astype(np.int32)
        frames = frames[indices]
        frames_indices = frames_indices[indices]

    metadata = VideoMetadata(
        total_num_frames=total_num_frames,
        fps=video_fps,
        frames_indices=frames_indices,
    )
    return frames, metadata


def read_video_frames(
    video: Union[str, List[str]],
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    fps: Optional[float] = None,
    max_frames: Optional[float] = None,
    **kwargs,
):
    if isinstance(video, str):
        frames = sorted([os.path.join(video, x) for x in os.listdir(video) if x.endswith((".jpg", ".jpeg", ".png"))])
    else:
        frames = video

    total_num_frames = len(frames)
    # if "shareVideoGPTV" in video:
    #     video_fps = 2
    # else:
    #     raise ValueError(f"Unkown video data source: {video}")
    video_fps = 2
    timestamps = [i / video_fps for i in range(total_num_frames)]
    frames_indices = list(range(total_num_frames))

    if start_time is not None:
        assert start_time >= 0, f"start_time {start_time} must be non-negative"
        start_index = min(range(len(timestamps)), key=lambda i: abs(timestamps[i] - start_time))
    else:
        start_index = 0

    if end_time is not None:
        assert end_time >= 0, f"end_time {end_time} must be non-negative"
        end_index = min(range(len(timestamps)), key=lambda i: abs(timestamps[i] - end_time))
        frames = frames[: end_index + 1]
        timestamps = timestamps[: end_index + 1]
    else:
        end_index = total_num_frames - 1

    frames_indices = frames_indices[start_index : end_index + 1]

    if fps is not None:
        assert fps <= video_fps, f"Cannot sample {fps} from {video_fps}"
        sample_rate = int(video_fps / fps)
        frames_indices = frames_indices[::sample_rate]

    if max_frames is not None and len(frames_indices) > max_frames:
        frames_indices = [frames_indices[round(i)] for i in np.linspace(0, len(frames_indices) - 1, max_frames)]

    frames = [Image.open(frames[i]).convert("RGB") for i in frames_indices]
    metadata = VideoMetadata(
        total_num_frames=total_num_frames,
        fps=video_fps,
        frames_indices=frames_indices,
    )
    return frames, metadata


def load_video(
    video: str,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    fps: Optional[float] = None,
    max_frames: Optional[float] = None,
    precise_time: bool = False,
    verbose: bool = False,
):
    if isinstance(video, (list, tuple)) or os.path.isdir(video):
        return read_video_frames(
            video=video,
            start_time=start_time,
            end_time=end_time,
            fps=fps,
            max_frames=max_frames,
        )

    return read_video_ffmpeg(
        video=video,
        start_time=start_time,
        end_time=end_time,
        fps=fps,
        max_frames=max_frames,
        precise_time=precise_time,
        verbose=verbose,
    )


def load_multimodal_data(
    conversation: List[Dict[str, Any]],
    fps: int = 1,
    max_frames: Optional[int] = None,
):
    new_conversation = []
    for message in conversation:
        contents = []
        for content in message["content"]:
            new_content = {"type": content["type"]}
            if content["type"] == "image":
                new_content["image"] = load_image(content["image"])
            elif content["type"] == "video":
                new_content["video"] = load_video(content["video"], fps=fps, max_frames=max_frames)
            elif content["type"] == "text":
                new_content["text"] = content["text"]
            else:
                raise ValueError(f"Unsupported content type: {content['type']}")
            contents.append(new_content)
        new_conversation.append({"role": message["role"], "content": contents})
    return new_conversation


def cross_entropy_loss(
    hidden_states,
    lm_head,
    position_ids,
    labels,
    num_items_in_batch,
    **kwargs,
):
    training_args = get_args()
    batch_size = hidden_states.size(0)

    shift_hidden_states = hidden_states[..., :-1, :]
    shift_labels = labels[..., 1:]
    mask = shift_labels >= 0
    shift_hidden_states = shift_hidden_states[mask].contiguous()
    shift_labels = shift_labels[mask].contiguous()

    if mask.sum() == 0:
        print(f"Get labels={labels}. Found no sample to calculate loss!")
        pseudo_logits = lm_head(hidden_states[:, 0:1])
        loss = 0.0 * pseudo_logits.mean()
        return loss

    if num_items_in_batch is None:
        reduction = "mean"
        denominator = None

    elif training_args.loss_reduction_scope == "batch":
        reduction = "sum"
        denominator = num_items_in_batch

    elif training_args.loss_reduction_scope == "sequence":
        reduction = "none"

        if batch_size == 1:
            # NOTE: packed sequence
            if position_ids.ndim == 3:
                position_ids = position_ids[0]
            start_indices = torch.nonzero(position_ids[0] == 0)[:, 0]
            end_indices = F.pad(start_indices[1:], (0, 1), value=position_ids.size(1))
            batch_indices = torch.cat(
                [
                    torch.full(
                        (e - s,),
                        fill_value=i,
                        device=position_ids.device,
                        dtype=torch.long,
                    )
                    for i, (s, e) in enumerate(zip(start_indices, end_indices))
                ],
            ).unsqueeze(0)
        else:
            batch_indices = torch.arange(batch_size, device=position_ids.device)
            batch_indices = batch_indices.unsqueeze(1).expand(-1, hidden_states.size(1))

        shift_batch_indices = batch_indices[..., :-1]
        shift_batch_indices = shift_batch_indices[mask].contiguous()
        num_tokens = F.one_hot(shift_batch_indices).sum(dim=0)
        denominator = num_tokens[shift_batch_indices] * num_items_in_batch

    else:
        raise ValueError(f"Unknown reduction scope: {training_args.loss_reduction_scope}")

    if training_args.loss_implementation == "torch":
        shift_logits = lm_head(shift_hidden_states)
        loss = torch.nn.functional.cross_entropy(
            shift_logits.float(),
            shift_labels,
            reduction=reduction,
        )
    elif training_args.loss_implementation == "cce":
        from cut_cross_entropy import linear_cross_entropy

        loss = linear_cross_entropy(
            shift_hidden_states,
            lm_head.weight,
            shift_labels,
            bias=lm_head.bias,
            reduction=reduction,
            accum_e_fp32=True,
            accum_c_fp32=True,
        )
    else:
        raise ValueError(f"Unkown loss implementation: {training_args.loss_implementation}")

    if denominator is not None:
        loss = loss / denominator
        if loss.ndim > 0:
            loss = loss.sum()

    return loss


class AllToAllFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input_tensor: torch.Tensor,
        output_split_sizes: Optional[List[int]] = None,
        input_split_sizes: Optional[List[int]] = None,
        group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> torch.Tensor:
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes
        ctx.group = group

        world_size = torch.distributed.get_world_size(group)
        if output_split_sizes is None:
            assert input_tensor.size(0) % world_size == 0
            output_split_sizes = [input_tensor.size(0) // world_size] * world_size
        if input_split_sizes is None:
            assert input_tensor.size(0) % world_size == 0
            input_split_sizes = [input_tensor.size(0) // world_size] * world_size

        output_tensor = input_tensor.new_zeros((sum(output_split_sizes), *input_tensor.shape[1:]))

        torch.distributed.all_to_all_single(
            output_tensor,
            input_tensor,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
        )

        return output_tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_input = grad_output.new_zeros((sum(ctx.input_split_sizes), *grad_output.shape[1:]))

        torch.distributed.all_to_all_single(
            grad_input,
            grad_output,
            output_split_sizes=ctx.input_split_sizes,
            input_split_sizes=ctx.output_split_sizes,
            group=ctx.group,
        )

        return grad_input, None, None, None


class EncoderLoadBalancingHandler(object):
    def __init__(
        self,
        grid_thw: torch.Tensor,
        merge_size: int = 1,
    ):
        self.group = get_encoder_load_balancing_group()

        self._activated = self.group is not None and get_args().encoder_load_balancing
        self.cu_seqlens = None
        if not self._activated:
            return

        self.world_size = torch.distributed.get_world_size(self.group)
        self.rank = torch.distributed.get_rank(self.group)

        num_tokens = grid_thw[:, 1:].prod(dim=1).repeat_interleave(grid_thw[:, 0])
        num_frames_ranks = [num_tokens.new_empty(1) for _ in range(self.world_size)]
        num_frames = num_tokens.new_ones(1) * len(num_tokens)
        torch.distributed.all_gather(
            num_frames_ranks,
            num_frames,
            group=self.group,
        )
        num_frames_ranks = [x.item() for x in num_frames_ranks]
        src_group_ids = [i for i, n in enumerate(num_frames_ranks) for _ in range(n)]

        if len(src_group_ids) <= self.world_size:
            self._activated = False
            return

        num_tokens_ranks = [num_tokens.new_empty(n) for n in num_frames_ranks]
        torch.distributed.all_gather(num_tokens_ranks, num_tokens, group=self.group)
        num_tokens_all = torch.cat(num_tokens_ranks).tolist()

        input_split_sizes, output_split_sizes, cu_seqlens = self._minimax_sum_split(num_tokens_all, src_group_ids)

        merge_factor = merge_size**2
        self.input_split_sizes = input_split_sizes
        self.output_split_sizes = output_split_sizes
        self.final_input_split_sizes = [x // merge_factor for x in output_split_sizes]
        self.final_output_split_sizes = [x // merge_factor for x in input_split_sizes]
        self.cu_seqlens = cu_seqlens

    @property
    def activated(self):
        return self._activated

    def _minimax_sum_split(
        self,
        num_tokens: List[int],
        src_group_ids: List[int],
    ):
        assert self.world_size <= len(num_tokens)

        def can_split(max_s):
            splits = 1
            current_sum = 0
            for num in num_tokens:
                if current_sum + num > max_s:
                    splits += 1
                    current_sum = num
                else:
                    current_sum += num
            return splits <= self.world_size

        left = max(num_tokens)
        right = sum(num_tokens)

        while left < right:
            mid = (left + right) // 2
            if can_split(mid):
                right = mid
            else:
                left = mid + 1

        limit = left

        input_split_sizes = [0] * self.world_size
        output_split_sizes = [0] * self.world_size
        cu_seqlens = [0]

        tgt_group_id = 0
        current_sum = 0

        for i, (num, src_group_id) in enumerate(zip(num_tokens, src_group_ids)):
            if current_sum + num > limit and tgt_group_id < self.world_size - 1:
                tgt_group_id += 1
                current_sum = num
            else:
                current_sum += num

            if src_group_id == self.rank:
                input_split_sizes[tgt_group_id] += num
            if tgt_group_id == self.rank:
                output_split_sizes[src_group_id] += num
                cu_seqlens.append(cu_seqlens[-1] + num)

            remaining_items = len(num_tokens) - 1 - i
            remaining_groups = self.world_size - 1 - tgt_group_id
            if remaining_items == remaining_groups and remaining_groups > 0:
                tgt_group_id += 1
                current_sum = 0

        return input_split_sizes, output_split_sizes, cu_seqlens

    def preprocess(self, hidden_states: torch.Tensor):
        if not self._activated:
            return hidden_states
        hidden_states = AllToAllFunction.apply(
            hidden_states,
            self.output_split_sizes,
            self.input_split_sizes,
            self.group,
        )
        return hidden_states

    def postprocess(self, hidden_states: torch.Tensor):
        if not self._activated:
            return hidden_states
        hidden_states = AllToAllFunction.apply(
            hidden_states,
            self.final_output_split_sizes,
            self.final_input_split_sizes,
            self.group,
        )
        return hidden_states


def binary_focal_loss_with_logits(logits, target, alpha=0.25, gamma=2.0, reduction='mean'):
    """
    logits: [bs, 1]，未sigmoid
    target: [bs, 1]  0或1
    """
    target = target.float()
    # BCE with logits loss本身就是 numerically stable的
    bce_loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none')  # [bs, 1]
    # 转概率
    prob = torch.sigmoid(logits)
    pt = prob * target + (1 - prob) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    focal_weight = alpha_t * (1 - pt) ** gamma

    loss = focal_weight * bce_loss
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

HF_HUB_PREFIX = 'hf-hub:'

def load_checkpoint_with_prefix(filename, prefix=None, map_location='cpu', logger='current'):
    """Load partial pretrained model with specific prefix.

    Args:
        prefix (str): The prefix of sub-module.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str | None): Same as :func:`torch.load`.
            Defaults to None.
        logger: logger

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    if filename.startswith('hf-hub:'):
        model_id = filename[len(HF_HUB_PREFIX):]
        filename = hf_hub_download(model_id, 'pytorch_model.bin')

    checkpoint = torch.load(filename, map_location=map_location)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    if not prefix:
        return state_dict
    if not prefix.endswith('.'):
        prefix += '.'
    prefix_len = len(prefix)

    state_dict = {
        k[prefix_len:]: v
        for k, v in state_dict.items() if k.startswith(prefix)
    }

    assert state_dict, f'{prefix} is not in the pretrained model'
    return state_dict


def load_state_dict_to_model(model, state_dict,  logger='current'):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict)
    if missing_keys:
        # print_log(missing_keys, logger=logger, level=logging.ERROR)
        raise RuntimeError()
    if unexpected_keys:
        # print_log(unexpected_keys, logger=logger, level=logging.ERROR)
        raise RuntimeError()

def genetate_video_pred_embeddings(pred_embeddings_list, frames_per_batch):

    pred_embeddings_list_video = []
    for pred_embedding_batch in pred_embeddings_list:
        pred_embeddings_list_video += [pred_embedding_batch] * frames_per_batch
    return pred_embeddings_list_video

def process_video_gt_masks(gt_masks, num_frames, mask_ids):
    gt_masks_processed = []
    mask_ids_flatten = [idx for idxs in mask_ids for idx in idxs]
    num_objs = len(mask_ids_flatten)
    for i in range(num_frames):
        for j in range(num_objs):
            idx = mask_ids_flatten[j]
            gt_masks_processed.append(gt_masks[idx][i:i+1])
    return gt_masks_processed

def dice_loss(pred,
              target,
              weight=None,
              eps=1e-3,
              reduction='mean',
              naive_dice=False,
              avg_factor=None):
    """Calculate dice loss, there are two forms of dice loss is supported:

        - the one proposed in `V-Net: Fully Convolutional Neural
            Networks for Volumetric Medical Image Segmentation
            <https://arxiv.org/abs/1606.04797>`_.
        - the dice loss in which the power of the number in the
            denominator is the first power instead of the second
            power.

    Args:
        pred (torch.Tensor): The prediction, has a shape (n, *)
        target (torch.Tensor): The learning label of the prediction,
            shape (n, *), same shape of pred.
        weight (torch.Tensor, optional): The weight of loss for each
            prediction, has a shape (n,). Defaults to None.
        eps (float): Avoid dividing by zero. Default: 1e-3.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        naive_dice (bool, optional): If false, use the dice
                loss defined in the V-Net paper, otherwise, use the
                naive dice loss in which the power of the number in the
                denominator is the first power instead of the second
                power.Defaults to False.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """

    input = pred.flatten(1)
    target = target.flatten(1).float()

    a = torch.sum(input * target, 1)
    if naive_dice:
        b = torch.sum(input, 1)
        c = torch.sum(target, 1)
        d = (2 * a + eps) / (b + c + eps)
    else:
        b = torch.sum(input * input, 1) + eps
        c = torch.sum(target * target, 1) + eps
        d = (2 * a) / (b + c)

    loss = 1 - d
    if weight is not None:
        assert weight.ndim == loss.ndim
        assert len(weight) == len(pred)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


# @MODELS.register_module()
class DiceLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 activate=True,
                 reduction='mean',
                 naive_dice=False,
                 loss_weight=1.0,
                 eps=1e-3):
        """Compute dice loss.

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            activate (bool): Whether to activate the predictions inside,
                this will disable the inside sigmoid operation.
                Defaults to True.
            reduction (str, optional): The method used
                to reduce the loss. Options are "none",
                "mean" and "sum". Defaults to 'mean'.
            naive_dice (bool, optional): If false, use the dice
                loss defined in the V-Net paper, otherwise, use the
                naive dice loss in which the power of the number in the
                denominator is the first power instead of the second
                power. Defaults to False.
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
            eps (float): Avoid dividing by zero. Defaults to 1e-3.
        """

        super(DiceLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.naive_dice = naive_dice
        self.loss_weight = loss_weight
        self.eps = eps
        self.activate = activate

    def forward(self,
                pred,
                target,
                weight=None,
                reduction_override=None,
                avg_factor=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction, has a shape (n, *).
            target (torch.Tensor): The label of the prediction,
                shape (n, *), same shape of pred.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction, has a shape (n,). Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        if self.activate:
            if self.use_sigmoid:
                pred = pred.sigmoid()
            else:
                raise NotImplementedError

        loss = self.loss_weight * dice_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            naive_dice=self.naive_dice,
            avg_factor=avg_factor)

        return loss



def reduce_loss(loss: Tensor, reduction: str) -> Tensor:
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss: Tensor,
                       weight: Optional[Tensor] = None,
                       reduction: str = 'mean',
                       avg_factor: Optional[float] = None) -> Tensor:
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Optional[Tensor], optional): Element-wise weights.
            Defaults to None.
        reduction (str, optional): Same as built-in losses of PyTorch.
            Defaults to 'mean'.
        avg_factor (Optional[float], optional): Average factor when
            computing the mean of losses. Defaults to None.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            # Avoid causing ZeroDivisionError when avg_factor is 0.0,
            # i.e., all labels of an image belong to ignore index.
            eps = torch.finfo(torch.float32).eps
            loss = loss.sum() / (avg_factor + eps)
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def cross_entropy(pred,
                  label,
                  weight=None,
                  reduction='mean',
                  avg_factor=None,
                  class_weight=None,
                  ignore_index=-100,
                  avg_non_ignore=False):
    """Calculate the CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (int | None): The label index to be ignored.
            If None, it will be set to default value. Default: -100.
        avg_non_ignore (bool): The flag decides to whether the loss is
            only averaged over non-ignored targets. Default: False.

    Returns:
        torch.Tensor: The calculated loss
    """
    # The default value of ignore_index is the same as F.cross_entropy
    ignore_index = -100 if ignore_index is None else ignore_index
    # element-wise losses
    loss = F.cross_entropy(
        pred,
        label,
        weight=class_weight,
        reduction='none',
        ignore_index=ignore_index)

    # average loss over non-ignored elements
    # pytorch's official cross_entropy average loss over non-ignored elements
    # refer to https://github.com/pytorch/pytorch/blob/56b43f4fec1f76953f15a627694d4bba34588969/torch/nn/functional.py#L2660  # noqa
    if (avg_factor is None) and avg_non_ignore and reduction == 'mean':
        avg_factor = label.numel() - (label == ignore_index).sum().item()

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def _expand_onehot_labels(labels, label_weights, label_channels, ignore_index):
    """Expand onehot labels to match the size of prediction."""
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    valid_mask = (labels >= 0) & (labels != ignore_index)
    inds = torch.nonzero(
        valid_mask & (labels < label_channels), as_tuple=False)

    if inds.numel() > 0:
        bin_labels[inds, labels[inds]] = 1

    valid_mask = valid_mask.view(-1, 1).expand(labels.size(0),
                                               label_channels).float()
    if label_weights is None:
        bin_label_weights = valid_mask
    else:
        bin_label_weights = label_weights.view(-1, 1).repeat(1, label_channels)
        bin_label_weights *= valid_mask

    return bin_labels, bin_label_weights, valid_mask


def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None,
                         class_weight=None,
                         ignore_index=-100,
                         avg_non_ignore=False):
    """Calculate the binary CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, 1) or (N, ).
            When the shape of pred is (N, 1), label will be expanded to
            one-hot format, and when the shape of pred is (N, ), label
            will not be expanded to one-hot format.
        label (torch.Tensor): The learning label of the prediction,
            with shape (N, ).
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (int | None): The label index to be ignored.
            If None, it will be set to default value. Default: -100.
        avg_non_ignore (bool): The flag decides to whether the loss is
            only averaged over non-ignored targets. Default: False.

    Returns:
        torch.Tensor: The calculated loss.
    """
    # The default value of ignore_index is the same as F.cross_entropy
    ignore_index = -100 if ignore_index is None else ignore_index

    if pred.dim() != label.dim():
        label, weight, valid_mask = _expand_onehot_labels(
            label, weight, pred.size(-1), ignore_index)
    else:
        # should mask out the ignored elements
        valid_mask = ((label >= 0) & (label != ignore_index)).float()
        if weight is not None:
            # The inplace writing method will have a mismatched broadcast
            # shape error if the weight and valid_mask dimensions
            # are inconsistent such as (B,N,1) and (B,N,C).
            weight = weight * valid_mask
        else:
            weight = valid_mask

    # average loss over non-ignored elements
    if (avg_factor is None) and avg_non_ignore and reduction == 'mean':
        avg_factor = valid_mask.sum().item()

    # weighted element-wise losses
    weight = weight.float()
    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), pos_weight=class_weight, reduction='none')
    # do the reduction for the weighted loss
    loss = weight_reduce_loss(
        loss, weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def mask_cross_entropy(pred,
                       target,
                       label,
                       reduction='mean',
                       avg_factor=None,
                       class_weight=None,
                       ignore_index=None,
                       **kwargs):
    """Calculate the CrossEntropy loss for masks.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C, *), C is the
            number of classes. The trailing * indicates arbitrary shape.
        target (torch.Tensor): The learning label of the prediction.
        label (torch.Tensor): ``label`` indicates the class label of the mask
            corresponding object. This will be used to select the mask in the
            of the class which the object belongs to when the mask prediction
            if not class-agnostic.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (None): Placeholder, to be consistent with other loss.
            Default: None.

    Returns:
        torch.Tensor: The calculated loss

    Example:
        >>> N, C = 3, 11
        >>> H, W = 2, 2
        >>> pred = torch.randn(N, C, H, W) * 1000
        >>> target = torch.rand(N, H, W)
        >>> label = torch.randint(0, C, size=(N,))
        >>> reduction = 'mean'
        >>> avg_factor = None
        >>> class_weights = None
        >>> loss = mask_cross_entropy(pred, target, label, reduction,
        >>>                           avg_factor, class_weights)
        >>> assert loss.shape == (1,)
    """
    assert ignore_index is None, 'BCE loss does not support ignore_index'
    # TODO: handle these two reserved arguments
    assert reduction == 'mean' and avg_factor is None
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    return F.binary_cross_entropy_with_logits(
        pred_slice, target, weight=class_weight, reduction='mean')[None]


# @MODELS.register_module()
class CrossEntropyLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 ignore_index=None,
                 loss_weight=1.0,
                 avg_non_ignore=False):
        """CrossEntropyLoss.

        Args:
            use_sigmoid (bool, optional): Whether the prediction uses sigmoid
                of softmax. Defaults to False.
            use_mask (bool, optional): Whether to use mask cross entropy loss.
                Defaults to False.
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            class_weight (list[float], optional): Weight of each class.
                Defaults to None.
            ignore_index (int | None): The label index to be ignored.
                Defaults to None.
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
            avg_non_ignore (bool): The flag decides to whether the loss is
                only averaged over non-ignored targets. Default: False.
        """
        super(CrossEntropyLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.ignore_index = ignore_index
        self.avg_non_ignore = avg_non_ignore
        if ((ignore_index is not None) and not self.avg_non_ignore
                and self.reduction == 'mean'):
            warnings.warn(
                'Default ``avg_non_ignore`` is False, if you would like to '
                'ignore the certain label and average loss over non-ignore '
                'labels, which is the same with PyTorch official '
                'cross_entropy, set ``avg_non_ignore=True``.')

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = cross_entropy

    def extra_repr(self):
        """Extra repr."""
        s = f'avg_non_ignore={self.avg_non_ignore}'
        return s

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=None,
                **kwargs):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The prediction.
            label (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The method used to reduce the
                loss. Options are "none", "mean" and "sum".
            ignore_index (int | None): The label index to be ignored.
                If not None, it will override the default value. Default: None.
        Returns:
            torch.Tensor: The calculated loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if ignore_index is None:
            ignore_index = self.ignore_index

        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(
                self.class_weight, device=cls_score.device)
        else:
            class_weight = None
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            ignore_index=ignore_index,
            avg_non_ignore=self.avg_non_ignore,
            **kwargs)
        return loss_cls



def dice_coefficient(x, target):
    """
    Dice Loss: 1 - 2 * (intersection(A, B) / (A^2 + B^2))
    :param x:
    :param target:
    :return:
    """
    eps = 1e-5
    n_inst = x.size(0)
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)
    intersection = (x * target).sum(dim=1)
    union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
    loss = 1. - (2 * intersection / union)
    return loss


def projection_loss(mask_scores, gt_bitmasks):
    mask_losses_y = dice_coefficient(
        mask_scores.max(dim=1, keepdim=True)[0],
        gt_bitmasks.max(dim=1, keepdim=True)[0]
    )
    mask_losses_x = dice_coefficient(
        mask_scores.max(dim=2, keepdim=True)[0],
        gt_bitmasks.max(dim=2, keepdim=True)[0]
    )
    return (mask_losses_x + mask_losses_y).mean()

def dice_score(pred_mask, gt_mask, eps=1e-6):
    """
    pred_mask: (H, W), sigmoid 后
    gt_mask:   (H, W), 0/1
    return: scalar in [0,1]
    """
    inter = (pred_mask * gt_mask).sum()
    union = pred_mask.sum() + gt_mask.sum()
    return (2 * inter + eps) / (union + eps)

def calculate_mask_loss_group(pred_masks, gt_masks, loss_mask, loss_dice, group_size=10):
    mask_bce_loss_ = 0
    mask_dice_loss_ = 0
    total_masks = 0
    for i in range(0, pred_masks.size(0), group_size):
        pred_chunk = pred_masks[i:i + group_size]
        gt_chunk = gt_masks[i:i + group_size]

        num_masks = pred_chunk.size(0)

        mask_bce_loss_ += loss_mask(pred_chunk, gt_chunk) * num_masks
        mask_dice_loss_ += loss_dice(pred_chunk, gt_chunk) * num_masks
        total_masks += num_masks

    return mask_bce_loss_, mask_dice_loss_, total_masks

import torch
import torch.nn.functional as F

def resize_pred_and_gt_for_loss(
    pred_masks: torch.Tensor,
    gt_mask: torch.Tensor,
    max_side: int = 1024,
    pred_mode: str = "bilinear",
    gt_mode: str = "nearest",
    align_corners: bool = False,
):
    assert pred_masks.dim() == 4, f"pred_masks must be 4D, got {pred_masks.shape}"
    assert gt_mask.dim() == 4, f"gt_mask must be 4D, got {gt_mask.shape}"
    if pred_masks.shape[-2:] == gt_mask.shape[-2:]:
        return pred_masks, gt_mask

    device = pred_masks.device
    dtype = pred_masks.dtype

    new_h, new_w = pred_masks.shape[-2:]
   
    # pred_resized = F.interpolate(
    #     pred_masks.float(),
    #     size=(new_h, new_w),
    #     mode=pred_mode,
    #     align_corners=align_corners if pred_mode != "nearest" else None,
    # ).to(device=device, dtype=dtype)

    with torch.no_grad():
        gt_resized = F.interpolate(
            gt_mask.float(),
            size=(new_h, new_w),
            mode=gt_mode,
        ).to(device=device)

    return pred_masks, gt_resized


def downsample_to_max_hw(pred_masks, gt_masks, max_h=1024, max_w=1024):
    """
    pred_masks: [N,1,H,W] (logits 或 prob 都行)
    gt_masks:   [N,1,H,W] (0/1 或 bool)
    """
    H, W = pred_masks.shape[-2:]
    # 不超阈值就不动
    if H <= max_h and W <= max_w:
        return pred_masks, gt_masks

    # 保持长宽比缩放到 <= max_h/max_w
    scale = min(max_h / H, max_w / W)

    pred_ds = F.interpolate(
        pred_masks, scale_factor=scale,
        mode="bilinear", align_corners=False
    )
    gt_ds = F.interpolate(
        gt_masks.float(), scale_factor=scale,
        mode="nearest"
    )
    return pred_ds, gt_ds
    
if __name__=='__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    start_id, end_id = 151646, 151647
    max_len = 4
    d = 256

    input_ids = torch.tensor(
        [999, start_id, 10, 11, 12, end_id, 888, start_id, 20, end_id, 777],
        device=device
    )

    n = input_ids.numel()
    hidden_states = torch.arange(n, device=device, dtype=torch.float32).unsqueeze(1).repeat(1, d)

    phrase_embeds, attn_mask, spans = get_phrase_embedding(input_ids, hidden_states, start_id, end_id, max_len=32)
    import pdb 
    pdb.set_trace()