import argparse
import os
import sys
sys.path.append('./')
from instructsam.models import load_pretrained_model


def resolve_model_path(path):
    if not os.path.isdir(path):
        return path

    checkpoints = []
    for name in os.listdir(path):
        if not name.startswith("checkpoint-"):
            continue
        step = name.removeprefix("checkpoint-")
        if step.isdigit():
            checkpoints.append((int(step), os.path.join(path, name)))

    if not checkpoints:
        return path

    return max(checkpoints, key=lambda item: item[0])[1]


parser = argparse.ArgumentParser()
parser.add_argument("--base_dir", type=str, default='./work_dirs')
parser.add_argument("--model_path", type=str, default='stage1')
parser.add_argument("--save_path", type=str, default='stage1_merged')
args = parser.parse_args()

save_path = os.path.join(args.base_dir, args.save_path)
model_path = resolve_model_path(os.path.join(args.base_dir, args.model_path))
print("Merging model from ", model_path)
tokenizer, model, processor = load_pretrained_model(model_path, None, attn_implementation='sdpa', save_path=save_path)

print("Model saved to ", save_path)
