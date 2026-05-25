import pathlib
import torch
from transformers.trainer_utils import enable_full_determinism, set_seed
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoImageProcessor,
    AutoVideoProcessor,
    AutoTokenizer,
    CONFIG_MAPPING,
    MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING,
    PROCESSOR_MAPPING,
)

from instructsam.training import (
    get_args,
    SFTDataset,
    TrainingArguments,
    DataCollator,
    Trainer,
    rank0_print,
    check_chat_template,
    find_all_linear_names
)
import sys
sys.path.append('./')
from instructsam.models.instructsam import InstructSAMForConditionalGeneration
from instructsam.constants import (REGION_TOKEN, SEG_TOKEN, REF_START_TOKEN, REF_END_TOKEN, SEG_START_TOKEN, SEG_END_TOKEN)
from instructsam.models.attention_ import *

def set_seed(seed=42):
    """
    Set the random seed for reproducible results.

    :param seed: An integer value to be used as the random seed.
    """
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_model(args: TrainingArguments):
    dtype = torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)

    original_config = AutoConfig.from_pretrained(args.model_path)
    enable_full_determinism(args.seed) if args.full_determinism else set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    image_processor = AutoImageProcessor.from_pretrained(args.model_path)
    video_processor = AutoVideoProcessor.from_pretrained(
        args.model_path,
        use_token_compression=args.use_token_compression,
    )
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        tokenizer=tokenizer,
        image_processor=image_processor,
        video_processor=video_processor,
    )

    processor.tokenizer.add_tokens([REGION_TOKEN], special_tokens=True)
    processor.tokenizer.add_tokens([SEG_TOKEN, REF_START_TOKEN, REF_END_TOKEN, SEG_START_TOKEN, SEG_END_TOKEN], special_tokens=True)

    original_config.region_token_index = processor.tokenizer.convert_tokens_to_ids(REGION_TOKEN)
    original_config.seg_token_index = processor.tokenizer.convert_tokens_to_ids(SEG_TOKEN)
    original_config.seg_start_token_index = processor.tokenizer.convert_tokens_to_ids(SEG_START_TOKEN)
    original_config.seg_end_token_index = processor.tokenizer.convert_tokens_to_ids(SEG_END_TOKEN)
    original_config.ref_start_token_index = processor.tokenizer.convert_tokens_to_ids(REF_START_TOKEN)
    original_config.ref_end_token_index = processor.tokenizer.convert_tokens_to_ids(REF_END_TOKEN)


    original_config.max_seg_nums = args.max_seg_nums
    original_config.seg_encoder = args.seg_encoder
    original_config.seg_decoder = args.seg_decoder
    original_config.mask_decoder_model = args.mask_decoder_model
    original_config.dice_loss_weight = 0.5
    original_config.bce_loss_weight = 2.0
    original_config.cls_loss_weight = 1.0
    original_config.loss_sample_points = args.loss_sample_points

    model = InstructSAMForConditionalGeneration.from_pretrained(
        args.model_path,
        config=original_config,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
    )


    if args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=args.lora_dropout,
            bias=args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if args.bits == 16:
            if args.bf16:
                model.to(torch.bfloat16)
            if args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if args.mask_decoder_model is not None:
        seg_processor = AutoProcessor.from_pretrained(args.mask_decoder_model)
    else:
        seg_processor = None

    if args.mask_decoder_model is not None and 'mm_mask_decoder' not in model.get_model().config:
        print('initialize mask decoder...')
        model.get_model().initialize_mask_decoder(model.get_model().config)

    # for p in model.get_model().parameters():
    #     p.requires_grad = True

    if args.llm_lr is None or args.llm_lr==0:
        for p in model.get_model().language_model.parameters():
            p.requires_grad = False

    if args.vision_encoder_lr is None or args.vision_encoder_lr==0:
        for p in model.get_model().visual.parameters():
            p.requires_grad = False

    if args.projector_lr is None or args.projector_lr==0:
        for p in model.get_model().visual.merger.parameters():
            p.requires_grad = False
    else:
        for p in model.get_model().visual.merger.parameters():
            p.requires_grad = True

    if args.sam_decoder_lr is None or args.sam_decoder_lr==0:
        for p in model.get_model().grounding_model.model.parameters():
            p.requires_grad = False
    else:
        for p in model.get_model().grounding_model.model.parameters():
            p.requires_grad = True

    if args.sam_encoder_lr is None or args.sam_encoder_lr==0:
        for p in model.get_model().grounding_model.model.vision_encoder.parameters():
            p.requires_grad = False
    else:
        for p in model.get_model().grounding_model.model.vision_encoder.parameters():
            p.requires_grad = False

    for n, p in model.named_parameters():
        if any(
            [
                x in n
                for x in ["lm_head", "embed_tokens", "text_hidden_fcs", "mask_hidden_fcs", "mask_queries"]
            ]
        ):
            # print(n)
            p.requires_grad = True
            
        if args.mask_queries_grad is False and "mask_queries" in n:
            p.requires_grad = False
    # print('requires grad params:')
    # for name, p in model.named_parameters():
    #     if p.requires_grad:
    #         print(name)
    # print('*****************')
    # import pdb 
    # pdb.set_trace()
    check_chat_template(processor)

    def mark_language_model_modules(model):
        for name, m in model.named_modules():
            if name.startswith("language_model."):
                setattr(m, "_is_in_language_model", True)

    mark_language_model_modules(model)

    return model, processor, seg_processor


def train():
    set_seed(42)
    args = get_args()

    model, processor, seg_processor = build_model(args)

    train_dataset = SFTDataset(
        model_config=model.config,
        processor=processor,
        seg_processor=seg_processor,
        model_max_length=args.model_max_length,
        mm_max_length=args.mm_max_length,
        fps=args.fps,
        max_frames=args.max_frames,
        dataloader_num_workers=args.dataloader_num_workers,
        data_args=args,
        requires_length=args.dynamic_batching or args.decoder_load_balancing,
        use_multi_objs=args.use_multi_objs
    )

    rank0_print(
        f"Model config: {model.config}\n\nModel: {model}\n\nProcessor: {processor}\n\n"
    )

    data_collator = DataCollator(
        processor=processor,
        sequence_packing=args.sequence_packing,
    )

    trainer = Trainer(
        model=model,
        args=args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        processing_class=processor,
    )


    resume_from_checkpoint = len(list(pathlib.Path(args.output_dir).glob("checkpoint-*"))) > 0
    return trainer.train(resume_from_checkpoint=resume_from_checkpoint)


if __name__ == "__main__":
    train()
