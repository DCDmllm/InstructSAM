import math
from collections.abc import Iterator
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.utils.data import Sampler
from tqdm import tqdm
from transformers import Trainer as TransformersTrainer
from transformers.trainer_utils import seed_worker
from transformers.trainer import has_length
from transformers.utils import (
    logging,
    is_sagemaker_mp_enabled,
)

from .utils import TrainingArguments

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp


logger = logging.get_logger(__name__)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['visual', 'text_hidden_fcs', 'grounding_model', 'class_head', 'mask_hidden_fcs', 'mask_queries']
    target_keywords = ["q_proj", "k_proj", "v_proj", "o_proj"]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if any(k in name for k in target_keywords):
            lora_module_names.add(name)
    
    # if 'lm_head' in lora_module_names: # needed for 16-bit
    #     lora_module_names.remove('lm_head')
    return list(lora_module_names)

def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks

def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class Trainer(TransformersTrainer):
    args: TrainingArguments

    def _get_train_sampler(self, train_dataset) -> Optional[torch.utils.data.Sampler]:
        if train_dataset is None:
            train_dataset = self.train_dataset
        if train_dataset is None or not has_length(train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler(train_dataset)

   
    def create_optimizer(self):
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = set(self.get_decay_parameter_names(opt_model))
            optimizer_grouped_parameters = []

            llm_parameters, projector_parameters, vision_encoder_parameters, mask_encoder_parameters, mask_decoder_parameters = [], [], [], [], []
            for n, p in opt_model.named_parameters():
                if "merger" in n:
                    projector_parameters.append((n, p))
                elif ("vision_model" in n or "visual" in n) and "grounding_model" not in n and "visual.merger" not in n:
                    vision_encoder_parameters.append((n, p))
                elif "grounding_model.model.vision_encoder" in n:
                    mask_encoder_parameters.append((n, p))
                elif "grounding_model" in n or "text_hidden_fcs" in n or "mask_hidden_fcs" in n or "mask_queries" in n:
                    mask_decoder_parameters.append((n, p))
                else:
                    llm_parameters.append((n, p))

            if self.args.llm_lr is not None and self.args.llm_lr > 0:
                optimizer_grouped_parameters.extend(
                    [
                        {
                            "name": "llm",
                            "params": [p for n, p in llm_parameters if (n in decay_parameters and p.requires_grad)],
                            "lr": self.args.llm_lr,
                            "weight_decay": self.args.weight_decay,
                        },
                        {
                            "name": "llm_nodecay",
                            "params": [
                                p for n, p in llm_parameters if (n not in decay_parameters and p.requires_grad)
                            ],
                            "lr": self.args.llm_lr,
                            "weight_decay": 0.0,
                        },
                    ]
                )

            if self.args.projector_lr is not None and self.args.projector_lr > 0:
                optimizer_grouped_parameters.extend(
                    [
                        {
                            "name": "projector",
                            "params": [
                                p for n, p in projector_parameters if (n in decay_parameters and p.requires_grad)
                            ],
                            "lr": self.args.projector_lr,
                            "weight_decay": self.args.weight_decay,
                        },
                        {
                            "name": "projector_nodecay",
                            "params": [
                                p for n, p in projector_parameters if (n not in decay_parameters and p.requires_grad)
                            ],
                            "lr": self.args.projector_lr,
                            "weight_decay": 0.0,
                        },
                    ]
                )

            if self.args.vision_encoder_lr is not None and self.args.vision_encoder_lr > 0:
                optimizer_grouped_parameters.extend(
                    [
                        {
                            "name": "vision_encoder",
                            "params": [
                                p for n, p in vision_encoder_parameters if (n in decay_parameters and p.requires_grad)
                            ],
                            "lr": self.args.vision_encoder_lr,
                            "weight_decay": self.args.weight_decay,
                        },
                        {
                            "name": "vision_encoder_nodecay",
                            "params": [
                                p for n, p in vision_encoder_parameters
                                if (n not in decay_parameters and p.requires_grad)
                            ],
                            "lr": self.args.vision_encoder_lr,
                            "weight_decay": 0.0,
                        },
                    ]
                )

            if self.args.sam_encoder_lr is not None and self.args.sam_encoder_lr > 0:
                optimizer_grouped_parameters.extend(
                    [
                        {
                            "name": "sam_encoder",
                            "params": [
                                p for n, p in mask_encoder_parameters if (n in decay_parameters and p.requires_grad)
                            ],
                            "lr": self.args.sam_encoder_lr,
                            "weight_decay": self.args.weight_decay,
                        },
                        {
                            "name": "sam_encoder_nodecay",
                            "params": [
                                p for n, p in mask_encoder_parameters
                                if (n not in decay_parameters and p.requires_grad)
                            ],
                            "lr": self.args.sam_encoder_lr,
                            "weight_decay": 0.0,
                        },
                    ]
                )

            if self.args.sam_decoder_lr is not None and self.args.sam_decoder_lr > 0:
                optimizer_grouped_parameters.extend(
                    [
                        {
                            "name": "sam_decoder",
                            "params": [
                                p for n, p in mask_decoder_parameters if (n in decay_parameters and p.requires_grad)
                            ],
                            "lr": self.args.sam_decoder_lr,
                            "weight_decay": self.args.weight_decay,
                        },
                        {
                            "name": "sam_decoder_nodecay",
                            "params": [
                                p for n, p in mask_decoder_parameters
                                if (n not in decay_parameters and p.requires_grad)
                            ],
                            "lr": self.args.sam_decoder_lr,
                            "weight_decay": 0.0,
                        },
                    ]
                )

            if self.optimizer_cls_and_kwargs is not None:
                optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
            else:
                optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)

            # Overwrite `params` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for GaLore optimizer.
            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")

            # Overwrite `model` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for LOMO optimizer.
            if "model" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("model")

            # For layer-wise dummy optimizers we overwrite optimizer_grouped_parameters with `optimizer_dict`
            # to avoid arguments conflicts.
            if "optimizer_dict" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            if "bitsandbytes" in str(optimizer_cls) and optimizer_kwargs.get("optim_bits", None) == 8:
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped / 2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped / 2**20}M params")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

    def update_history_loss_dict(self,outputs):
        if not hasattr(self,'history_loss_dict'):
            self.history_loss_dict = {}
        for name, value in outputs.items():
            if 'loss' in name and name != 'loss':
                if name not in self.history_loss_dict:
                    self.history_loss_dict[name] = value.item()
                else:
                    if value != 0:
                        self.history_loss_dict[name] = value.item()

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ):
        pc = getattr(self.accelerator, "parallelism_config", None)
        if pc is not None and pc.sp_backend == "deepspeed" and pc.sp_enabled:
            return self._deepspeed_sp_compute_loss(model, inputs, return_outputs, pc)

        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if self.model_accepts_loss_kwargs:
            kwargs = {}
            if num_items_in_batch is not None:
                kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **kwargs}
        outputs = model(**inputs)

        # User-defined compute_loss function
        if self.compute_loss_func is not None:
            if labels is None:
                logger.warning(
                    "Trainer: `compute_loss_func` is defined but `labels=None`. "
                    "Your custom loss function will still be called with labels=None. "
                )
            loss = self.compute_loss_func(
                outputs,
                labels,
                num_items_in_batch=num_items_in_batch,
            )
        # Default HF loss handling (label smoothing) if no custom loss function
        elif labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            model_name = (
                unwrapped_model.base_model.model._get_name()
                if _is_peft_model(unwrapped_model)
                else unwrapped_model._get_name()
            )
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            if isinstance(outputs, dict) and 'mask_loss' in outputs:
                loss_dict = {}
                for name,value in outputs.items():
                    if 'loss' in name and name != 'loss':
                        loss_value = value.item()
                        if loss_value == 0 and hasattr(self,'history_loss_dict'):
                            loss_value = self.history_loss_dict[name]
                        loss_dict[name] = loss_value
                self.update_history_loss_dict(outputs)
                self.log(loss_dict)

        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes if self.args.n_gpu <= 1 else self.args.n_gpu
        return (loss, outputs) if return_outputs else loss

    def _save_checkpoint(self, model, trial):
        if self.args.lora_enable:
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            state_dict = get_peft_state_maybe_zero_3(self.model.named_parameters(), self.args.lora_bias)
            non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(self.model.named_parameters())
        
            # add for qwen2
            if hasattr(self.model, 'base_model') and hasattr(self.model.base_model, 'lm_head'):
                lm_head_weight = self.model.base_model.lm_head.weight.cpu() 
                non_lora_state_dict['base_model.lm_head.weight'] = lm_head_weight
                print("add base_model.lm_head.weight")
            else:
                print("The model does not have 'base_model.lm_head.weight' attribute.")


            if self.args.local_rank == 0 or self.args.local_rank == -1:
                # save for acquring `config.json`
                self.model.config.save_pretrained(output_dir)
                # save for acquring `adapter_config.json`, `adapter_model.bin`
                # self.model.save_pretrained(output_dir, state_dict=state_dict)
                torch.save(non_lora_state_dict, os.path.join(output_dir, 'non_lora_trainables.bin'))
            super(Trainer, self)._save_checkpoint(model, trial)
        else:
            super(Trainer, self)._save_checkpoint(model, trial)