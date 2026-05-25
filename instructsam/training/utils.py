from dataclasses import dataclass, field
from typing import List, Optional
import random

import torch
from transformers import (
    HfArgumentParser,
    TrainingArguments as TransformersTrainingArguments,
)


__all__ = [
    "TrainingArguments",
    "get_args",
    "rank0_print",
    "check_chat_template",
    "get_encoder_load_balancing_group",
]

_ARGS = None

_ENCODER_LOAD_BALANCING_GROUP = None


@dataclass
class TrainingArguments(TransformersTrainingArguments):
    # Model arguments
    model_path: Optional[str] = field(default=None)
    model_type: Optional[str] = field(default=None)
    vision_encoder_path: Optional[str] = field(default=None)
    attn_implementation: Optional[str] = field(default="sdpa")
    use_token_compression: Optional[bool] = field(default=False)
    use_liger_kernel: bool = field(default=False)

    # Data arguments
    ann_path: List[str] = field(default=None)
    data_root: Optional[str] = field(default=None)
    data_path_root: Optional[str] = field(default='/')
    data_cache_dir: Optional[str] = field(default=None)

    model_max_length: Optional[int] = field(default=16384)
    mm_max_length: Optional[int] = field(default=10240)
    fps: Optional[int] = field(default=1)
    max_frames: Optional[int] = field(default=180)

    # Training arguments
    llm_lr: float = field(default=2e-5)
    projector_lr: float = field(default=2e-5)
    vision_encoder_lr: float = field(default=2e-5)
    sam_encoder_lr: Optional[float] = None
    sam_decoder_lr: Optional[float] = 8e-5

    sequence_packing: bool = field(default=False)
    decoder_load_balancing: bool = field(default=False)
    encoder_load_balancing: bool = field(default=False)
    encoder_load_balancing_size: int = field(default=4)
    dynamic_batching: bool = field(default=False)
    dynamic_batching_window_size: int = field(default=128)

    loss_implementation: str = field(default="torch")
    loss_reduction_scope: str = field(default="sequence")
    average_tokens_across_devices: bool = field(default=True)

    group_by_modality_length: bool = field(default=False)
    loss_sample_points: bool = field(default=False)

    use_multi_objs: bool = field(default=False)
    skip_none: bool = field(default=True)

    # Lora or Quant Arguments
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"

    # Segmentation arguments
    max_seg_nums: int = field(default=10)
    seg_encoder: str = field(default="sam3")
    seg_decoder: str = field(default="sam3")
    mask_decoder_model: Optional[str] = field(default="checkpoints/sam3")
    mask_queries_grad: bool = field(default=True)

    def __post_init__(self):
        super().__post_init__()
        assert self.model_path is not None

        assert self.loss_implementation in ["torch", "cce"], (
            f"Unsupported loss implementation: {self.loss_implementation}"
        )
        if self.loss_implementation == "cce":
            try:
                import cut_cross_entropy  # noqa: F401
            except ImportError as e:
                raise ImportError(f"Failed to import `cut_cross_entropy`: {e}")

        if self.sequence_packing:
            assert "flash_attention" in self.attn_implementation, "Sequence packing requires flash attention."

        if self.decoder_load_balancing:
            assert self.sequence_packing, "DP load balancing requires batch flattening."
            assert not self.dynamic_batching, "DP load balancing and dynamic batching cannot be used together."
            assert not self.group_by_length, "DP load balancing and group by length cannot be used together."

        if self.dynamic_batching:
            assert self.sequence_packing, "Dynamic batching requires batch flattening."
            assert not self.decoder_load_balancing, "Dynamic batching and workload balancing cannot be used together."
            assert not self.group_by_length, "Dynamic batching and group by length cannot be used together."

        if self.use_liger_kernel:
            try:
                import liger_kernel  # noqa: F401
            except ImportError as e:
                raise ImportError(f"Failed to import `liger_kernel`: {e}")

        assert self.loss_reduction_scope in ["batch", "sequence"], (
            f"Unsupported loss reduction scope: {self.loss_reduction_scope}"
        )
        if self.loss_reduction_scope == "sequence":
            assert self.average_tokens_across_devices

        if self.encoder_load_balancing:
            assert torch.distributed.is_initialized()
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()

            assert self.encoder_load_balancing_size > 1
            assert self.encoder_load_balancing_size <= world_size

            groups = torch.arange(world_size).view(-1, self.encoder_load_balancing_size)
            for group in groups:
                pg = torch.distributed.new_group(
                    ranks=group.tolist(),
                    backend="nccl",
                )
                if rank in group:
                    global _ENCODER_LOAD_BALANCING_GROUP
                    _ENCODER_LOAD_BALANCING_GROUP = pg


def get_encoder_load_balancing_group():
    return _ENCODER_LOAD_BALANCING_GROUP


def get_args() -> TrainingArguments:
    global _ARGS
    if _ARGS is None:
        parser = HfArgumentParser(TrainingArguments)
        _ARGS = parser.parse_args_into_dataclasses()[0]
    return _ARGS


def rank0_print(*args, **kwargs):
    if torch.distributed.get_rank() == 0:
        print(*args, **kwargs)


def check_chat_template(processor) -> bool:
    conversation = [
        {"role": "user", "content": [{"type": "text", "text": "Hello!"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Hello!"}]},
    ]
    prompt = processor.tokenizer.apply_chat_template(
        conversation, tokenize=False, chat_template=processor.chat_template
    )
    prompt_local = processor.apply_chat_template(conversation, tokenize=False)
    assert prompt == prompt_local, "Chat template in local implementation does not match the processor."

def clean_phrase(phrase):
    phrase = phrase.strip()
    phrase = phrase.lower()[0]+phrase[1:]
    if phrase.endswith('.'):
        phrase = phrase[:-1]  
    return phrase

SEG_IMAGE_QUESTIONS_PHRASE = [
    "Please segment '{phrase}'.",
    "Can you segment '{phrase}' in this image?",
    "Please segment '{phrase}' in this image.",
    "Could you provide a segmentation mask for '{phrase}' in this image?",
    "Please identify and segment '{phrase}' in this image.",
    "Where is '{phrase}' in this image? Please respond with a segmentation mask.",
    "Can you highlight '{phrase}' in this image with a segmentation mask?",
]

SEG_IMAGE_QUESTIONS_PHRASE_MULTI = [
    "Please segment '{phrase1} and {phrase2}'.",
    "Can you segment '{phrase1} and {phrase2}' in this image?",
    "Please segment '{phrase1} and {phrase2}' in this image.",
    "Could you provide a segmentation mask for '{phrase1} and '{phrase2}' in this image?",
    "Please identify and segment '{phrase1} and {phrase2}' in this image.",
    "Where is '{phrase1} and {phrase2}' in this image? Please respond with a segmentation mask.",
    "Can you highlight '{phrase1} and {phrase2}' in this image with a segmentation mask?",

]

SEG_VIDEO_QUESTIONS_PHRASE = [
    "Can you segment '{phrase}' in this video?",
    "Please segment '{phrase}' in this video.",
    "Could you provide a segmentation mask for '{phrase}' in this video?",
    "Please identify and segment '{phrase}' in this video.",
    "Where is '{phrase}' in this video? Please respond with a segmentation mask.",
    "Can you highlight '{phrase}' in this video with a segmentation mask?",

]

SEG_IMAGE_QUESTIONS_OCR = [
    "Please segment the text '{phrase}'.",
    "Can you segment the text '{phrase}' in this image?",
    "Please segment the text '{phrase}' in this image.",
    "Could you provide a segmentation mask for the text '{phrase}' in this image?",
    "Please identify and segment the text '{phrase}' in this image.",
    "Where is the text '{phrase}' in this image? Please respond with a segmentation mask.",
    "Can you highlight the text '{phrase}' in this image with a segmentation mask?",
]
