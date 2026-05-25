import argparse

import torch
import sys
sys.path.append('./')
from instructsam.models import load_pretrained_model
from instructsam import mm_infer_segmentation
from PIL import Image
import torch.nn.functional as F
import numpy as np
from pathlib import Path

def infer_and_vis(query, image_path, processor, model, tokenizer, output_dir):
    contents = []
    contents.append({"type": "image", "image": image_path})
    contents.append({"type": "text", "text": query})

    conversation = [{"role": "user", "content": contents}]

    output, masks, cls_score = mm_infer_segmentation(
        image_path,
        processor,
        conversation,
        model,
        tokenizer
    )
    print(output)

    if masks is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        original_image = Image.open(image_path).convert("RGBA")
        width, height = original_image.size   

        alpha_value = 128  # 0~255，数值越大越不透明

        for i in range(masks.shape[1]):
            mask = masks[:,i:i+1]

            pred_masks = F.interpolate(
                mask.float(), 
                size=(height, width), 
                mode='bilinear', 
                align_corners=False
            )

            # 二值化
            pred_masks = (pred_masks > 0)

            # 转成 0/255 的 uint8
            pred_mask_np = pred_masks[0, 0].detach().cpu().numpy().astype(np.uint8) * 255

            # 转成 PIL 灰度图，再转 RGBA
            pil_mask = Image.fromarray(pred_mask_np, mode="L")

            color_overlay = Image.new("RGBA", (width, height), (255, 0, 0, 0))  # 红色，可改成其他颜色
            # 把灰度 mask 缩放透明度到 alpha_value
            alpha_mask = pil_mask.point(lambda p: int(p > 0) * alpha_value)

            # 把带透明度的颜色层叠加到原图上
            overlay = Image.new("RGBA", (width, height), (255, 0, 0, 0))
            overlay.putalpha(alpha_mask)

            combined = Image.alpha_composite(original_image, overlay)
            safe_query = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in query)[:80]
            combined.save(output_dir / f'{safe_query}_{i}.png')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='work_dirs/stage2')
    parser.add_argument("--query", type=str, default="Please segment 'the man on the left' in the image.") #person, tree, car, road, tv, cup, table and phone
    parser.add_argument("--image-path", type=str, default='assets/desert.jpg')
    parser.add_argument("--output-dir", type=str, default='vis')
    args = parser.parse_args()

    tokenizer, model, processor = load_pretrained_model(args.model_path, None, attn_implementation='sdpa')

    model.to(torch.bfloat16)
    query = args.query
    image_path = args.image_path

    infer_and_vis(query, image_path, processor, model, tokenizer, args.output_dir)
    

    

if __name__ == "__main__":
    main()
