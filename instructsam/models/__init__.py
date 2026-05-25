from . import qwen3_vl

import torch
import os
from .instructsam import InstructSAMForConditionalGeneration
from transformers import (
    AutoConfig,
    PretrainedConfig,
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
from peft import PeftConfig
from safetensors.torch import load_file
from .attention_ import *

def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-') or 'bak' in model_paths[-1]:
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]

def load_pretrained_model(model_path, model_base, load_8bit=False, load_4bit=False, device_map="auto", **kwargs):
    model_name = get_model_name_from_path(model_path)
    if 'token' in kwargs:
        token = kwargs['token']
    else:
        token = None
    
    save_path = kwargs.pop('save_path', False)

    # NOTE: auto device_map by default
    # if want to put model into a single device, you can set device_map={"": "cuda:0"}
    kwargs = {"device_map": device_map, **kwargs}

    config = AutoConfig.from_pretrained(model_path)
    config._attn_implementation = kwargs.pop('attn_implementation', "flash_attention_2") # default to flash_attention_2

    torch_dtype = config.torch_dtype if hasattr(config, "torch_dtype") else kwargs.pop('torch_dtype', torch.float16)

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        # NOTE: High-version Transformers will report: """ValueError: You can't pass `load_in_4bit`or `load_in_8bit` as a kwarg when passing `quantization_config` argument at the same time."""
        # kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch_dtype
    # judge model type
    model_type = config.model_type if hasattr(config, "model_type") else kwargs.pop('model_type', "qwen3_vl")

    # judge pretrain/finetune
    is_alignment = getattr(config, "tune_mm_mlp_adapter", False) or getattr(config, "is_alignment", False)

    # NOTE: lora/qlora model loading
    if 'lora' in model_name.lower() or 'qlora' in model_name.lower():
    # if True:
        cfg_pretrained = PeftConfig.from_pretrained(model_path, token=token)
        # NOTE: AutoConfig will modify `_name_or_path` property to `model_path` if `model_path` is not None.
        # cfg_pretrained = AutoConfig.from_pretrained(model_path, token=token)
        model_base = model_base if model_base is not None else cfg_pretrained.base_model_name_or_path

        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, token=token)
        print('Loading Qwen from base model...')
        print(model_base)
        
        model = InstructSAMForConditionalGeneration.from_pretrained(
            model_base, 
            low_cpu_mem_usage=True, 
            config=config, 
            ignore_mismatched_sizes=True, 
            attn_implementation="fused_attention",
            **kwargs
        )

        print('Loading additional Qwen3 weights...')
        if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
            non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu', weights_only=True,)
        else:
            # this is probably from HF Hub
            from huggingface_hub import hf_hub_download
            def load_from_hf(repo_id, filename, subfolder=None):
                cache_file = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    subfolder=subfolder)
                return torch.load(cache_file, map_location='cpu')
            non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')

        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
        if any(k.startswith('model.model.') for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
    
        non_lora_frozen = load_file(os.path.join(config.mask_decoder_model, 'model.safetensors'))
        for k, v in non_lora_frozen.items():
            k = k.replace('detector_model', 'model.grounding_model.model')
            if k not in non_lora_trainables:
                non_lora_trainables[k] = v
        model.load_state_dict(non_lora_trainables, strict=False)

        from peft import PeftModel
        print('Loading LoRA weights...')
        model = PeftModel.from_pretrained(model, model_path)
        print('Merging LoRA weights...')
        model = model.merge_and_unload()
        print('Model is loaded...')


        def mark_language_model_modules(model):
            for name, m in model.named_modules():
                if name.startswith("language_model."):
                    setattr(m, "_is_in_language_model", True)

        mark_language_model_modules(model)



    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, token=token)
        model = InstructSAMForConditionalGeneration.from_pretrained(model_path, config=config, **kwargs)

    processor = AutoProcessor.from_pretrained(
        model_path,
    )
    

    if save_path:
        model.save_pretrained(save_path, state_dict=model.state_dict())
        tokenizer.save_pretrained(save_path)
        processor.save_pretrained(save_path)

    return tokenizer, model, processor
