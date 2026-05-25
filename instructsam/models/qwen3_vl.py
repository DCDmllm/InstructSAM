import math
from collections import defaultdict
from typing import Optional, Union, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers.models.qwen3_vl.video_processing_qwen3_vl import smart_resize
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.processing_qwen3_vl import (
    Qwen3VLProcessor as _Qwen3VLProcessor,
    Qwen3VLProcessorKwargs,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLModel as _Qwen3VLModel,
    Qwen3VLForConditionalGeneration as _Qwen3VLForConditionalGeneration,
    Qwen3VLVisionModel as _Qwen3VLVisionModel,
    Qwen3VLCausalLMOutputWithPast,
    Qwen3VLModelOutputWithPast,
    Qwen3VLTextModel,
)
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import prepare_fa_kwargs_from_position_ids
from transformers.processing_utils import AllKwargsForChatTemplate, Unpack, BatchFeature, MultiModalData
from transformers.utils import is_torchdynamo_compiling
from transformers.utils.generic import TransformersKwargs, check_model_inputs

from .utils import load_multimodal_data, cross_entropy_loss, EncoderLoadBalancingHandler


class Qwen3VLVisionModel(_Qwen3VLVisionModel):
    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> torch.Tensor:
        fake_forward = False
        if hidden_states.size(0) == 0:
            fake_forward = True
            hidden_states = hidden_states.new_zeros(
                (
                    self.spatial_merge_size * self.spatial_merge_size,
                    self.patch_size * self.patch_size * self.config.in_channels * self.config.temporal_patch_size,
                ),
            )
            grid_thw = grid_thw.new_tensor([[1, self.spatial_merge_size, self.spatial_merge_size]])

        hidden_states = self.patch_embed(hidden_states)

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)

        handler = EncoderLoadBalancingHandler(grid_thw=grid_thw, merge_size=self.spatial_merge_size)
        hidden_states = handler.preprocess(hidden_states)
        rotary_pos_emb = handler.preprocess(rotary_pos_emb)

        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        if handler.activated:
            cu_seqlens = grid_thw.new_tensor(handler.cu_seqlens, dtype=torch.int32)
        else:
            cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
                dim=0,
                # Select dtype based on the following factors:
                #  - FA2 requires that cu_seqlens_q must have dtype int32
                #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
                # See https://github.com/huggingface/transformers/pull/34852 for more information
                dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
            )
            cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            if layer_num in self.deepstack_visual_indexes:
                deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](
                    hidden_states
                )
                deepstack_feature = handler.postprocess(deepstack_feature)
                if fake_forward:
                    deepstack_feature = deepstack_feature[:0]
                deepstack_feature_lists.append(deepstack_feature)

        hidden_states = self.merger(hidden_states)
        hidden_states = handler.postprocess(hidden_states)
        if fake_forward:
            hidden_states = hidden_states[:0]

        return hidden_states, deepstack_feature_lists


class Qwen3VLModel(_Qwen3VLModel):
    def __init__(self, config: Qwen3VLConfig):
        super(_Qwen3VLModel, self).__init__(config)
        self.visual = Qwen3VLVisionModel._from_config(config.vision_config)
        self.language_model = Qwen3VLTextModel._from_config(config.text_config)
        self.rope_deltas = None  # cache rope_deltas here

        in_channels = self.config.vision_config.in_channels
        patch_size = self.config.vision_config.patch_size
        temporal_patch_size = self.config.vision_config.temporal_patch_size
        self.visual_in_channels = patch_size * patch_size * in_channels * temporal_patch_size

        # Initialize weights and apply final processing
        self.post_init()

    def get_multimodal_features(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
    ):
        if pixel_values is None:
            pixel_values = torch.zeros(0, self.visual_in_channels, dtype=self.visual.dtype, device=self.visual.device)
            image_grid_thw = torch.zeros((0, 3), dtype=torch.long, device=self.visual.device)

        if pixel_values_videos is None:
            pixel_values_videos = torch.zeros(
                0, self.visual_in_channels, dtype=self.visual.dtype, device=self.visual.device
            )
            video_grid_thw = torch.zeros((0, 3), dtype=torch.long, device=self.visual.device)

        pixel_values = torch.cat([pixel_values, pixel_values_videos], dim=0).type(self.visual.dtype)
        grid_thw = torch.cat([image_grid_thw, video_grid_thw], dim=0)
        visual_embeds, deepstack_visual_embeds = self.visual(pixel_values, grid_thw=grid_thw)

        num_image_tokens = image_grid_thw.prod(dim=1).sum() // self.visual.spatial_merge_size**2
        image_embeds = visual_embeds[:num_image_tokens]
        video_embeds = visual_embeds[num_image_tokens:]
        deepstack_image_embeds = [x[:num_image_tokens] for x in deepstack_visual_embeds]
        deepstack_video_embeds = [x[num_image_tokens:] for x in deepstack_visual_embeds]

        return image_embeds, video_embeds, deepstack_image_embeds, deepstack_video_embeds

    @check_model_inputs
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLModelOutputWithPast]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None

        if self.training or pixel_values is not None or pixel_values_videos is not None:
            image_embeds, video_embeds, deepstack_image_embeds, deepstack_video_embeds = self.get_multimodal_features(
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
            )

            image_mask, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        else:
            image_embeds = video_embeds = None
            deepstack_image_embeds = deepstack_video_embeds = None

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            # aggregate visual_pos_masks and deepstack_visual_embeds
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds

        if position_ids is None:
            attention_mask_tensor = (
                attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                # Only apply conversion for floating point tensors (inverted masks)
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            # Calculate RoPE index once per generation in the pre-fill stage only.
            # When compiling, we can't check tensor values thus we check only input length
            # It is safe to assume that `length!=1` means we're in pre-fill because compiled
            # models currently cannot do asssisted decoding
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask=attention_mask_tensor,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

        return Qwen3VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
        )


class Qwen3VLForConditionalGeneration(_Qwen3VLForConditionalGeneration):
    accepts_loss_kwargs = True

    def __init__(self, config):
        super(_Qwen3VLForConditionalGeneration, self).__init__(config)
        self.model = Qwen3VLModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        self.post_init()

    @check_model_inputs
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLCausalLMOutputWithPast]:
        if labels is not None:
            (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k) = prepare_fa_kwargs_from_position_ids(
                position_ids[0]
            )
            kwargs["cu_seq_lens_q"] = cu_seq_lens_q
            kwargs["cu_seq_lens_k"] = cu_seq_lens_k
            kwargs["max_length_q"] = max_length_q
            kwargs["max_length_k"] = max_length_k

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]

        loss, logits = None, None
        if labels is not None:
            loss = cross_entropy_loss(
                hidden_states=hidden_states,
                lm_head=self.lm_head,
                position_ids=position_ids,
                labels=labels,
                **kwargs,
            )
        else:
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(hidden_states[:, slice_indices, :])

        return Qwen3VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )


class Qwen3VLProcessor(_Qwen3VLProcessor):
    def apply_chat_template(
        self,
        conversation: List[Dict[str, str]],
        chat_template: Optional[str] = None,
        mm_max_length: Optional[int] = None,
        return_labels: bool = False,
        **kwargs: Unpack[AllKwargsForChatTemplate],
    ):
        if return_labels:
            assert kwargs.get("return_tensors", None) == "pt", (
                "`return_tensors` must be set to `pt` when `return_labels` is True."
            )
            assert not kwargs.get("add_generation_prompt", False), (
                "`add_generation_prompt` must be set to False when `return_labels` is True."
            )
            assert kwargs.get("tokenize", True), "`tokenize` must be set to True when `return_labels` is True."
            assert kwargs.get("return_dict", False), "`return_dict` must be set to True when `return_labels` is True."

            pseudo_message = [{"role": "user", "content": [{"type": "text", "text": ""}]}]
            prompt_tokens = super().apply_chat_template(
                pseudo_message, chat_template=chat_template, tokenize=True, add_generation_prompt=False
            )[0]
            conv_tokens = super().apply_chat_template(
                pseudo_message, chat_template=chat_template, tokenize=True, add_generation_prompt=True
            )[0]
            prompt_length = len(conv_tokens) - len(prompt_tokens)

            ignore_tokens = torch.as_tensor(
                [self.image_token_id, self.video_token_id, self.vision_start_token_id, self.vision_end_token_id]
            )[None, None]

        fps = kwargs.pop("fps", 1)
        max_frames = kwargs.pop("max_frames", None)
        tokenize = kwargs.pop("tokenize", True)
        return_dict = kwargs.pop("return_dict", False)
        return_tensors = kwargs.pop("return_tensors", None)
        add_generation_prompt = kwargs.pop("add_generation_prompt", False)
        kwargs.pop("do_sample_frames", False)

        if tokenize and return_dict:
            conversation = load_multimodal_data(
                conversation,
                fps=fps,
                max_frames=max_frames,
            )

            if mm_max_length is not None:
                assert "max_pixels" not in kwargs and "size" not in kwargs, (
                    "Please provide only one of `mm_max_length` and `max_pixels`."
                )
                num_images, num_videos = 0, 0
                for message in conversation:
                    for content in message["content"]:
                        if content["type"] == "image":
                            num_images += 1
                        elif content["type"] == "video":
                            num_videos += 1
                kwargs["size"] = {
                    # FIXME: add an argument to control `shortest_edge`
                    "shortest_edge": self.image_processor.size["shortest_edge"],
                    "longest_edge": self._get_max_pixels(
                        num_images=num_images,
                        num_videos=num_videos,
                        mm_max_length=mm_max_length,
                    ),
                }

        outputs = defaultdict(list)

        for i, message in enumerate(conversation):
            prompt = super().apply_chat_template(
                [message],
                chat_template=chat_template,
                tokenize=False,
                add_generation_prompt=add_generation_prompt and i == len(conversation) - 1,
            )

            if tokenize and return_dict:
                images, videos, video_metadatas = [], [], []
                if message["role"] != "assistant":
                    for content in message["content"]:
                        if content["type"] == "image":
                            images.append(content["image"])
                        elif content["type"] == "video":
                            videos.append(content["video"][0])
                            video_metadatas.append(content["video"][1])

                results = self(
                    text=prompt,
                    images=images if len(images) > 0 else None,
                    videos=videos if len(videos) > 0 else None,
                    video_metadata=video_metadatas if len(videos) > 0 else None,
                    return_tensors="pt",
                    do_sample_frames=False,
                    **kwargs,
                )

                if return_labels:
                    labels = torch.full_like(results["input_ids"], fill_value=-100, dtype=torch.long)
                    if message["role"] == "assistant":
                        valid_mask = torch.all(results["input_ids"][..., None] != ignore_tokens, dim=-1)
                        # prefix: <|im_start|>assistant\n
                        valid_mask[:, :prompt_length] = False
                        # postfix: \n
                        valid_mask[:, -1] = False
                        labels[valid_mask] = results["input_ids"][valid_mask]
                    results["labels"] = labels

                for key, value in results.items():
                    outputs[key].append(value)

            else:
                outputs["prompts"].append(prompt)

        if tokenize:
            mm_input_names = set(self.image_processor.model_input_names + self.video_processor.model_input_names)
            for k, v in outputs.items():
                if k in mm_input_names:
                    outputs[k] = torch.cat(v, dim=0)
                else:
                    outputs[k] = torch.cat(v, dim=1)
            outputs = BatchFeature(outputs, tensor_type=return_tensors)
            if return_dict:
                return outputs
            return outputs["input_ids"]

        return "".join(outputs["prompts"])

    def _get_max_pixels(
        self,
        num_images: int,
        num_videos: int,
        mm_max_length: Optional[int] = None,
    ):
        merge_size = max(self.image_processor.merge_size, self.video_processor.merge_size)
        if num_images > 0:
            merge_size = min(merge_size, self.image_processor.merge_size)
        if num_videos > 0:
            merge_size = min(merge_size, self.video_processor.merge_size)
        factor = self.image_processor.patch_size * merge_size
        return mm_max_length // max(num_images + num_videos, 1) * (factor**2)

    def _get_number_of_video_patches(self, num_frames: int, height: int, width: int, videos_kwargs=None):
        min_pixels = videos_kwargs.get("min_pixels", None) or self.video_processor.size["shortest_edge"]
        max_pixels = videos_kwargs.get("max_pixels", None) or self.video_processor.size["longest_edge"]
        patch_size = videos_kwargs.get("patch_size", None) or self.video_processor.patch_size
        merge_size = videos_kwargs.get("merge_size", None) or self.video_processor.merge_size
        temporal_patch_size = (
            videos_kwargs.get("temporal_patch_size", None) or self.video_processor.temporal_patch_size
        )

        factor = patch_size * merge_size
        resized_height, resized_width = smart_resize(
            num_frames=num_frames,
            height=height,
            width=width,
            temporal_factor=temporal_patch_size,
            factor=factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
        grid_t = math.ceil(num_frames / temporal_patch_size)
        return grid_t * grid_h * grid_w

    def _get_num_multimodal_tokens(
        self,
        image_sizes=None,
        video_sizes=None,
        mm_max_length: Optional[int] = None,
        **kwargs,
    ):
        if mm_max_length is not None:
            assert "max_pixels" not in kwargs, "Please provide only one of `mm_max_length` and `max_pixels`."
            kwargs["max_pixels"] = self._get_max_pixels(
                num_images=len(image_sizes) if image_sizes is not None else 0,
                num_videos=len(video_sizes) if video_sizes is not None else 0,
                mm_max_length=mm_max_length,
            )

        vision_data = {}
        if image_sizes is not None:
            images_kwargs = Qwen3VLProcessorKwargs._defaults.get("images_kwargs", {})
            images_kwargs.update(kwargs)
            merge_size = images_kwargs.get("merge_size", None) or self.image_processor.merge_size

            num_image_patches = [
                self.image_processor.get_number_of_image_patches(*image_size, images_kwargs)
                for image_size in image_sizes
            ]
            num_image_tokens = [(num_patches // merge_size**2) for num_patches in num_image_patches]
            vision_data.update({"num_image_tokens": num_image_tokens, "num_image_patches": num_image_patches})

        if video_sizes is not None:
            videos_kwargs = Qwen3VLProcessorKwargs._defaults.get("videos_kwargs", {})
            videos_kwargs.update(kwargs)
            merge_size = videos_kwargs.get("merge_size", None) or self.video_processor.merge_size

            fps = kwargs.pop("fps", 1)
            max_frames = kwargs.pop("max_frames", None)
            for video_size in video_sizes:
                num_frames = video_size[0] // fps
                if max_frames is not None:
                    num_frames = min(num_frames, max_frames)
                video_size[0] = num_frames

            num_video_patches = [
                self._get_number_of_video_patches(*video_size, videos_kwargs) for video_size in video_sizes
            ]
            num_video_tokens = [(num_patches // merge_size**2) for num_patches in num_video_patches]
            vision_data["num_video_tokens"] = num_video_tokens

        return MultiModalData(**vision_data)


transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLForConditionalGeneration = Qwen3VLForConditionalGeneration
transformers.models.auto.modeling_auto.MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING[Qwen3VLConfig] = (
    Qwen3VLForConditionalGeneration
)

transformers.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor = Qwen3VLProcessor
transformers.models.auto.processing_auto.PROCESSOR_MAPPING[Qwen3VLConfig] = Qwen3VLProcessor
