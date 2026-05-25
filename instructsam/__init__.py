from . import models
from PIL import Image, ImageOps
import torch
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
)

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def mm_infer_segmentation(image_path, processor, conversation, model, tokenizer, **kwargs):
    seg_processor = AutoProcessor.from_pretrained(model.config.mask_decoder_model)

    # sam image
    sam_images = []
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    sam_inputs = seg_processor(image)
    sam_images.append(sam_inputs['pixel_values'][0])
    sam_size = sam_inputs.original_sizes[0]
    sam_images = torch.cat(sam_images, dim=0)

    # model inputs
    inputs = processor.apply_chat_template(
        conversation=conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    with torch.inference_mode():
        output_ids, pred_masks, cls_score = model.inference(
            **inputs,
            sam_images=[sam_images.to(model.device)],
            max_new_tokens=1024,
            use_cache=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
            do_sample=False
        )
    outputs = processor.tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0].strip()
    outputs = outputs.replace("<|object_ref_end|>", "<|object_ref_end|><|mask_start|>[SEG]<|mask_end|>")
    return outputs, pred_masks, cls_score