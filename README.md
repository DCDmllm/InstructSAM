<div align="center">

# InstructSAM: Segment Any Instance with Any Instructions

[![arXiv preprint](https://img.shields.io/badge/Arxiv-2510.23603-ECA8A7?logo=arxiv)](https://arxiv.org/abs/2510.23603)
[![Huggingface](https://img.shields.io/badge/Model-HuggingFace-E6A151)](https://huggingface.co/CircleRadon/InstructSAM-2B)
[![Data](https://img.shields.io/badge/Dataset-Huggingface-7EBDC2)](https://huggingface.co/datasets/CircleRadon/Inst2Seg)
[![Benchmark](https://img.shields.io/badge/Benchmark-Huggingface-B8A1E6)](https://huggingface.co/datasets/CircleRadon/Inst2Seg-Bench)

</div>


https://github.com/user-attachments/assets/39753d3c-c4ff-4fb9-af6f-d96baf63d45b

## Overview

InstructSAM is an instruction-driven multi-instance segmentation framework designed to segment arbitrary target instances from natural-language instructions.

<img src="assets/model.png" width="95%">

Key features:

- **Flexible instructions**: supports category prompts, referring expressions, and reasoning-style instructions.
- **Instance-aware outputs**: predicts a set of instance masks instead of a single semantic region.
- **Efficient inference**: avoids multi-round agentic prompting and repeated SAM calls.
- **Inst2Seg dataset support**: includes training and evaluation scripts for instruction-based instance segmentation.

## News

* **[2026.5.26]** 🔥 We release InstructSAM.

## Setup

Create a conda environment with Python 3.10:

```bash
conda create -n instructsam python=3.10 -y
conda activate instructsam
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

## Training
### Download data
We provide the training annotation JSON files on [Hugging Face](https://huggingface.co/datasets/CircleRadon/instructsam_training_eval_data). Download all JSON files and place them under `data/training`.

Raw images should be downloaded from the official source of each dataset.

### Stage 1

Stage 1 starts from the base [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) checkpoint:

```bash
bash scripts/train/stage1.sh
```

### Merge Stage 1 Checkpoint

After Stage 1 finishes, merge the Stage 1 LoRA weights into the base checkpoint:

```bash
python3 -m instructsam.merge_ckpt \
  --base_dir ./work_dirs \
  --model_path instructsam_stage1_2b \
  --save_path instructsam_stage1_merged
```


### Stage 2: Reasoning Fine-tuning

Stage 2 starts from the merged Stage 1 checkpoint:

```bash
bash scripts/train/stage2.sh
```

## Inference

### Download data
We provide the evaluation annotation JSON files on [Hugging Face](). Download all JSON files and place them under `data/eval`.

Run single-image inference:

```bash
python3 -m instructsam.infer \
  --model_path work_dirs/stage2 \
  --image-path path/to/image.jpg \
  --query "Please segment the object in the image." \
  --output-dir vis
```

The script prints the generated text and mask scores, then writes mask overlays to `vis/`.


## Evaluation

Available evaluation entry points:

```bash
bash evaluation/scripts/eval_inst2seg.sh
bash evaluation/scripts/eval_reasonseg.sh
bash evaluation/scripts/eval_grefcoco_ap.sh
bash evaluation/scripts/eval_roborefit.sh
```

Edit the dataset roots in each script before running. The Python evaluation files require explicit `--image_folder` and `--question_file` arguments instead of relying on machine-specific defaults.



## Citation

If you find this project useful, please cite using this BibTeX:
