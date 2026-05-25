import math
import inspect
import os
from collections import defaultdict
from typing import List, Optional, Tuple, Union, Dict, Any
from dataclasses import dataclass

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
    Qwen3VLTextModel,
)
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import prepare_fa_kwargs_from_position_ids
from transformers.processing_utils import AllKwargsForChatTemplate, Unpack, BatchFeature, MultiModalData
from transformers.utils import is_torchdynamo_compiling, can_return_tuple
from transformers.utils.generic import TransformersKwargs, check_model_inputs
from transformers.modeling_outputs import ModelOutput

from .utils import load_multimodal_data, cross_entropy_loss, EncoderLoadBalancingHandler, CrossEntropyLoss, DiceLoss, genetate_video_pred_embeddings, process_video_gt_masks, binary_focal_loss_with_logits, projection_loss, get_phrase_embedding, expand_vision_features, get_phrase_ids_by_start_end, dice_score, calculate_mask_loss_group, downsample_to_max_hw, select_vision_outputs, resize_pred_and_gt_for_loss
from .segmentation_decoder import SegmentationDecoder
from .assigner import HungarianAssigner
from instructsam.constants import IGNORE_INDEX
from .omni_attention import omni_attn_mask_naive, full_attn_mask, fused_full_attn_mask
from .point_sample import sample_points

@dataclass
class InstructSAMModelOutputWithPast(ModelOutput):

    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None
    labels: Optional[torch.LongTensor] = None

@dataclass
class InstructSAMCausalLMOutputWithPast(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    ce_loss: Optional[torch.FloatTensor] = None
    mask_bce_loss: Optional[torch.FloatTensor] = None
    mask_dice_loss: Optional[torch.FloatTensor] = None
    mask_loss: Optional[torch.FloatTensor] = None
    cls_loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None


class InstructSAMModel(_Qwen3VLModel): 
    def __init__(self, config: Qwen3VLConfig):
        super(InstructSAMModel, self).__init__(config)
        if 'out_dim' not in config:
            config.out_dim = 256    
        self.build_mask_decoder(config)
        self.grounding_model = SegmentationDecoder(config)   

    def initialize_mask_decoder(self, config):
        self.grounding_model.load_model(config)
        self.config.mm_mask_decoder = config.mask_decoder_model
        with torch.no_grad():
            self.mask_queries.zero_()

    def build_mask_decoder(self, config):
        # if config.training:
        # self.class_head = SegPresenceClassifier()
            
        # Projection layer for reasonseg
        in_dim = config.text_config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True    

        mask_fcs = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.mask_hidden_fcs = nn.ModuleList([nn.Sequential(*mask_fcs)])
        self.mask_hidden_fcs.train()
        for param in self.mask_hidden_fcs.parameters():
            param.requires_grad = True    

        self.mask_queries = nn.Parameter(torch.zeros(config.max_seg_nums, config.text_config.hidden_size))  

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
        masks: Optional[List[torch.LongTensor]] = None,
        mask_ids = None,
        sam_images = None,
        masks_valid = None,
        mask_type = None,
        labels = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, InstructSAMModelOutputWithPast]:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None

        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds, deepstack_video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

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

        # replace [SEG] token with queries
        if input_ids is not None:
            B, N = input_ids.shape
            mask_selected = (input_ids == self.config.seg_token_index)
            modality_batch = []
            # print(mask_selected.sum())
            
            if mask_selected.sum() > 0: 
                mask_num = mask_selected.sum()//self.config.max_seg_nums
                mask_feats = self.mask_queries.repeat(mask_num,1)
                inputs_embeds[mask_selected] = inputs_embeds[mask_selected]*0.0 + mask_feats

                mask_indices = mask_selected.nonzero(as_tuple=False)  # [n, 2] -> (b, pos)
                mask_indices_right = mask_indices.clone()
                mask_indices_right[:, 1] = mask_indices_right[:, 1] + 1
                valid = mask_indices_right[:, 1] < N
                mask_indices_right = mask_indices_right[valid]

                mask_selected_right = torch.zeros_like(mask_selected)
                mask_selected_right[mask_indices_right[:, 0], mask_indices_right[:, 1]] = True
                labels[mask_selected_right] = IGNORE_INDEX # 第一个[seg]算loss 最后的<mask_end>不算loss
                labels[mask_selected] = IGNORE_INDEX

                # get start and end idx for each [SEG]
                for b in range(B):
                    row = mask_selected[b] 

                    padded = F.pad(row, (1, 1), value=False)  # [N+2]
                    diff = padded[1:].to(torch.int8) - padded[:-1].to(torch.int8)

                    starts = torch.nonzero(diff == 1, as_tuple=False).squeeze(1)  # in [0, N]
                    ends   = torch.nonzero(diff == -1, as_tuple=False).squeeze(1) # in [0, N]

                    if starts.numel() == 0:
                        spans = torch.empty((0, 2), device=input_ids.device, dtype=torch.long)
                    else:
                        spans = torch.stack([starts, ends], dim=1)  # [num_spans, 2]

                    modality_batch.append(spans)

        if position_ids is None:
            past_key_values_length = 0 if past_key_values is None else past_key_values.get_seq_length()
            if self.rope_deltas is None or past_key_values_length == 0:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask=attention_mask,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (past_key_values_length + self.rope_deltas).to(inputs_embeds.device)
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        if attention_mask is not None and attention_mask.dim()==2:
            attention_mask = omni_attn_mask_naive(attention_mask, modality_batch)
        

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

        return InstructSAMModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
            labels=labels,
        )

class InstructSAMForConditionalGeneration(_Qwen3VLForConditionalGeneration):
    def __init__(self, config):
        super(_Qwen3VLForConditionalGeneration, self).__init__(config)
        self.model = InstructSAMModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

        self.loss_mask = CrossEntropyLoss(
            use_sigmoid=True,
            reduction='mean',
            loss_weight=2.0
        )
        self.loss_dice = DiceLoss(
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=0.5
        )

        self.assigner = HungarianAssigner(
            dice_loss_weight=config.dice_loss_weight,
            ce_loss_weight=config.bce_loss_weight,
            cls_loss_weight=config.cls_loss_weight,
        )

    def get_model(self):
        return self.model
    
    @can_return_tuple
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
        masks: Optional[List[torch.LongTensor]] = None,
        mask_ids = None,
        sam_images = None,
        masks_valid = None,
        mask_type = None,
        phrase_ids = None,
        data_indices = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, InstructSAMCausalLMOutputWithPast]:
        # print('input_ids', input_ids.shape)
        # print('pixel_values', pixel_values.shape)
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
            masks=masks,
            mask_ids=mask_ids,
            sam_images=sam_images,
            masks_valid=masks_valid,
            mask_type=mask_type,
            labels=labels,
            **kwargs,
        )

        hidden_states = outputs['last_hidden_state']

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        ce_loss = None
        mask_bce_loss = None
        mask_dice_loss = None
        mask_loss = None
        cls_loss = None

        if labels is not None: # training
            ce_loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)
            
            mask_valid_ = False
            if masks[0] is not None:
                hidden_states_sam = self.model.mask_hidden_fcs[0](hidden_states)
                g_pixel_values = torch.stack(sam_images, dim=0)  # [bs, C, H, W]
                with torch.no_grad():
                    vision_outputs_batch = self.model.grounding_model.encoder(g_pixel_values)

                bs = input_ids.shape[0]
                mask_bce_sum = None
                mask_dice_sum = None
                cls_sum = None
                num_masks = 0
                num_cls = 0
                for i in range(bs):
                    pred_masks = []
                    input_id = input_ids[i]
                    seg_token_mask = input_id==self.config.seg_token_index
                    
                    pred_embedding = hidden_states_sam[i][seg_token_mask]
                    pred_embedding = pred_embedding.reshape(-1, self.config.max_seg_nums, pred_embedding.shape[-1])
                    # print('pred_embedding shape:', pred_embedding.shape) # [num_seg, max_seg_nums, dim]

                    phrase_id = input_ids.new_tensor(phrase_ids[i])
                    phrase_embedding = self.model.get_input_embeddings()(phrase_id)
                    phrase_embedding = self.model.text_hidden_fcs[0](phrase_embedding.unsqueeze(0)).squeeze(0)
                
                    gt_mask = masks[i]
                    mask_valid_ = masks_valid[i]

                    g_pixel_values = sam_images[i].unsqueeze(0)

                    vision_outputs = select_vision_outputs(vision_outputs_batch, i)

                    obj_num = pred_embedding.shape[0]

                    max_chunk = 5
                    # print('obj_num:', obj_num, 'vision_outputs shape:', vision_outputs['last_hidden_state'].shape)

                    all_mask_outputs = []
                    pred_masks_list = []
                    pred_logits_list = []

                    phrase_embedding, text_attn_mask = get_phrase_embedding(
                        phrase_id, phrase_embedding, self.config.ref_start_token_index
                    )

                    if phrase_embedding.shape[0] == 0:
                        mask_valid_ = False
                        print(data_indices)
                        print('phrase_embedding is empty')
                        break
                    else:
                        for start in range(0, obj_num, max_chunk):
                            end = min(start + max_chunk, obj_num)
                            chunk_size = end - start

                            pred_embedding_chunk = pred_embedding[start:end]  # [chunk, max_seg_nums, dim]（按你原实现）
                            vision_outputs_expand = expand_vision_features(vision_outputs, chunk_size)

                            mask_outputs_chunk = self.model.grounding_model.decoder(
                                vision_outputs_expand,
                                phrase_embedding[start:end],
                                text_attn_mask[start:end],
                                pred_embedding_chunk
                            )

                            pred_masks_chunk = mask_outputs_chunk["pred_masks"]   # [chunk, 50, H, W]
                            pred_logits_chunk = mask_outputs_chunk["pred_logits"] # [chunk, 50]

                            # resize：只处理当前 chunk（gt_mask 返回的也可直接复用）
                            pred_masks_chunk, gt_mask_rs = resize_pred_and_gt_for_loss(pred_masks_chunk, gt_mask)

                            # 对 chunk 内每个对象做 assign + loss
                            for local_midx in range(chunk_size):
                                global_midx = start + local_midx  # 对齐你原来的 midx

                                mask_id = mask_ids[i][global_midx]
                                mask_id_tensor = torch.as_tensor(mask_id, device=pred_masks_chunk.device, dtype=torch.long)

                                # 注意：尽量别 .float()，除非你的 loss 必须 FP32
                                pred_masks_cur = pred_masks_chunk[local_midx].unsqueeze(1)  # [50, 1, H, W]
                                pred_scores_cur = pred_logits_chunk[local_midx]             # [50]
                                gt_masks_cur = gt_mask_rs[mask_id_tensor]                   # [Ng, 1, H, W]

                                if gt_mask_rs.sum()>0: # not null

                                    assign_id = self.assigner.assign(
                                        pred_masks_cur.float(),
                                        gt_masks_cur.float(),
                                        pred_scores_cur.float(),
                                    )

                                    score_targets = torch.zeros_like(pred_scores_cur)
                                    for id_, asid in enumerate(assign_id):
                                        if asid != -1:
                                            gt_masks_ = gt_masks_cur[asid:asid+1]      # [1,1,H,W]
                                            pred_masks_ = pred_masks_cur[id_:id_+1]     # [1,1,H,W]

                                            if mask_type[i] == 0:  # mask
                                                if self.config.loss_sample_points:
                                                    sampled_pred_mask, sampled_gt_mask = sample_points(pred_masks_, gt_masks_)
                                                    bce = self.loss_mask(sampled_pred_mask, sampled_gt_mask)
                                                    dice = self.loss_dice(sampled_pred_mask, sampled_gt_mask)
                                                else:
                                                    bce = self.loss_mask(pred_masks_, gt_masks_)
                                                    dice = self.loss_dice(pred_masks_, gt_masks_)
                                            elif mask_type[i] == 1:  # bbox
                                                dice = projection_loss(pred_masks_, gt_masks_)
                                                bce = dice * 0.0
                                            else:
                                                raise NotImplementedError

                                            # bce = bce * _scale
                                            # dice = dice * _scale
                                            if mask_bce_sum is None:
                                                mask_bce_sum = bce
                                                mask_dice_sum = dice
                                            else:
                                                mask_bce_sum = mask_bce_sum + bce
                                                mask_dice_sum = mask_dice_sum + dice
                                            num_masks += 1

                                            q_score = dice_score(pred_masks_cur[id_].sigmoid(), gt_masks_cur[asid])
                                            score_targets[id_] = max(q_score.item(), 0.1)
                                        else:
                                            score_targets[id_] = 0.0
                                else: # null gt
                                    score_targets = torch.zeros_like(pred_scores_cur)

                                cls_ = F.binary_cross_entropy_with_logits(pred_scores_cur, score_targets, reduction="mean")
                                if cls_sum is None:
                                    cls_sum = cls_
                                else:
                                    cls_sum = cls_sum + cls_
                                num_cls += 1

                            # del mask_outputs_chunk, pred_masks_chunk, pred_logits_chunk, vision_outputs_expand, pred_embedding_chunk

                if mask_bce_sum is not None:
                    mask_bce_loss = mask_bce_sum / num_masks
                    mask_dice_loss = mask_dice_sum / num_masks
                else:
                    mask_bce_loss = ce_loss*0.0
                    mask_dice_loss = ce_loss*0.0

                if cls_sum is not None:
                    cls_loss = cls_sum / num_cls
                else:
                    cls_loss = ce_loss*0.0
                
                mask_bce_loss = self.config.bce_loss_weight * mask_bce_loss 
                mask_dice_loss = self.config.dice_loss_weight * mask_dice_loss 
                cls_loss = self.config.cls_loss_weight * cls_loss
                mask_loss = mask_bce_loss + mask_dice_loss
                loss = mask_loss + ce_loss + cls_loss
    
            if not mask_valid_:
                # print('No valid masks found.')
                loss = ce_loss
                mask_bce_loss = loss * 0.0
                mask_dice_loss = loss * 0.0
                mask_loss = loss * 0.0
                cls_loss = loss * 0.0


        if mask_loss is not None: # training
            return InstructSAMCausalLMOutputWithPast(
                loss=loss,
                ce_loss=ce_loss.detach(),
                mask_bce_loss=mask_bce_loss.detach(),
                mask_dice_loss=mask_dice_loss.detach(),
                mask_loss=mask_loss.detach(),
                cls_loss=cls_loss.detach(),
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                rope_deltas=outputs.rope_deltas,
            )
        else:
            return InstructSAMCausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                rope_deltas=outputs.rope_deltas,
            )

    def inference(
        self,
        masks: Optional[List[torch.LongTensor]] = None,
        mask_ids = None,
        sam_images = None,
        masks_valid = None,
        mask_type = None,
        phrase_ids = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        self.SEG_START = None
        self.seg_output_embeddings = []
        outputs = self.generate(
            **kwargs
        )

        input_ids = kwargs['input_ids']
        output_ids = outputs.sequences
        # last_hidden_state = []
        # for hs in outputs.hidden_states: # round
        #     last_hidden_state.append(hs[-1])
        # last_hidden_state = torch.cat(last_hidden_state, dim=1)


        pred_masks = None
        pred_logits = None
        try:
            if len(self.seg_output_embeddings)>0:
                seg_output_embeddings = torch.cat(self.seg_output_embeddings, dim=0)
                pred_embeddings = self.model.mask_hidden_fcs[0](seg_output_embeddings)

                g_pixel_values = sam_images[0].unsqueeze(0)

                vision_outputs = self.model.grounding_model.encoder(g_pixel_values)

                obj_num = pred_embeddings.shape[0]
                
                vision_outputs_expand = expand_vision_features(vision_outputs, obj_num)

                phrase_id, text_attn_mask = get_phrase_ids_by_start_end(output_ids, self.config.ref_start_token_index, self.config.ref_end_token_index)

                phrase_embedding = self.model.get_input_embeddings()(phrase_id)
                phrase_embedding = self.model.text_hidden_fcs[0](phrase_embedding)

                mask_outputs = self.model.grounding_model.decoder(vision_outputs_expand, phrase_embedding, text_attn_mask, pred_embeddings)

                pred_masks = mask_outputs['pred_masks'] # [9,10,288,288]
                pred_logits = mask_outputs['pred_logits'].sigmoid()

        except Exception as exp:
            print('Segmentation inference error:', exp)
            print(seg_output_embeddings.shape)
            print(output_ids)
            pred_masks = None
            pred_logits = None
            
        
        output_ids = output_ids[:, input_ids.shape[1]:]
        return output_ids, pred_masks, pred_logits


    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Cache | None = None,
        attention_mask: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ):

        attention_mask_ = attention_mask

        # 1. Handle BC:
        model_inputs = {}
        model_inputs["cache_position"] = cache_position

        # 2. Generic cache-dependent input preparation
        if past_key_values is not None:
            model_inputs["past_key_values"] = past_key_values
        # We check `use_cache` below because some stateful models (like `recurrent_gemma`) expect input slicing if
        # their caching mechanism is used. To define `use_cache`, the user-defined argument takes precedence.
        use_cache = kwargs.get("use_cache")
        if use_cache is None:
            use_cache = getattr(self.config, "use_cache", False)
        if past_key_values is not None or use_cache:
            # TODO (joao): handle the case where cache length == input_ids length. The function below results in an
            # exception because we get empty input_ids after slicing. In essence, we need to roll back the cache 1
            # token to recompute the logits for the first token to be generated (but not all caches support roll backs)
            inputs_embeds, input_ids = self._cache_dependant_input_preparation(
                input_ids, inputs_embeds, cache_position
            )

        # 3. Prepare base model inputs
        input_ids_key = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step for every prompt.
        if not self.config.is_encoder_decoder:
            if inputs_embeds is not None and len(cache_position) == inputs_embeds.shape[1]:
                model_inputs[input_ids_key] = None
                model_inputs["inputs_embeds"] = inputs_embeds
            else:
                if not hasattr(self.config, "seg_start_token_index"):
                    self.config.seg_start_token_index = 151671

                if self.SEG_START=='1': # add <mask_end>
                    self.SEG_START = None
                #     # print('before:',input_ids)
                #     # print('Add <mask_end> token for segment generation!')
                #     input_ids[0][0] = self.config.seg_end_token_index
                #     model_inputs[input_ids_key] = input_ids.clone(memory_format=torch.contiguous_format)
                #     model_inputs["inputs_embeds"] = None
                #     self.SEG_START = None

                if input_ids[0][0] == self.config.ref_end_token_index: # replace [SEG] token
                    self.SEG_START = '1'
                    # print('Segment token detected in generation, use mask queries!')
                    model_inputs[input_ids_key] = None
                    ref_end_embedding = self.model.get_input_embeddings()(
                        torch.tensor([self.config.ref_end_token_index], dtype=torch.long, device=self.model.device)
                    ).unsqueeze(0)

                    seg_start_embedding = self.model.get_input_embeddings()(
                        torch.tensor([self.config.seg_start_token_index], dtype=torch.long, device=self.model.device)
                    ).unsqueeze(0)

                    seg_end_embedding = self.model.get_input_embeddings()(
                        torch.tensor([self.config.seg_end_token_index], dtype=torch.long, device=self.model.device)
                    ).unsqueeze(0)
                            
                    model_inputs["inputs_embeds"] = torch.cat(
                        [ref_end_embedding, seg_start_embedding, self.model.mask_queries.unsqueeze(0), seg_end_embedding], dim=1
                    )
                    
                    attention_mask_ = torch.cat(
                        [attention_mask_, attention_mask_.new_ones((1, self.config.max_seg_nums+3))], dim=-1)
                
                else:
                    # print('input_ids', input_ids)
                    # `clone` calls in this function ensure a consistent stride. See #32227
                    model_inputs[input_ids_key] = input_ids.clone(memory_format=torch.contiguous_format)
                    model_inputs["inputs_embeds"] = None
                
        else:
            model_inputs[input_ids_key] = input_ids.clone(memory_format=torch.contiguous_format)

        # 4. Create missing `position_ids` on the fly
        encoder_attention_mask = attention_mask if self.config.is_encoder_decoder else None
        attention_mask = (
            kwargs.pop("decoder_attention_mask", None) if self.config.is_encoder_decoder else attention_mask
        )
        attention_mask_key = "decoder_attention_mask" if self.config.is_encoder_decoder else "attention_mask"
        position_ids_key = "decoder_position_ids" if self.config.is_encoder_decoder else "position_ids"
        if (
            attention_mask_ is not None
            and kwargs.get(position_ids_key) is None
            and position_ids_key in set(inspect.signature(self.forward).parameters.keys())
        ):
            position_ids = attention_mask_.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask_ == 0, 1)
            kwargs[position_ids_key] = position_ids  # placed in kwargs for further processing (see below)

        # 5. Slice model inputs if it's an input that should have the same length as `input_ids`
        for model_input_name in ["position_ids", "token_type_ids", "decoder_position_ids"]:
            model_input = kwargs.get(model_input_name)
            if model_input is not None:
                if past_key_values is not None or use_cache:
                    current_input_length = (
                        model_inputs["inputs_embeds"].shape[1]
                        if model_inputs.get("inputs_embeds") is not None
                        else model_inputs[input_ids_key].shape[1]
                    )
                    model_input = model_input[:, -current_input_length:]
                    model_input = model_input.clone(memory_format=torch.contiguous_format)
                model_inputs[model_input_name] = model_input

        # 6. Create 4D attention mask is we are using a compilable cache (important for performant compiled forward
        # pass)
        if (
            isinstance(past_key_values, Cache)
            and past_key_values.is_compileable
            and attention_mask_ is not None
            and attention_mask_.ndim == 2
        ):
            if not self.config.is_encoder_decoder and model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
            else:
                batch_size, sequence_length = model_inputs[input_ids_key].shape[:2]

            # Create the causal mask with fixed shape in advance, to reduce recompilations. If the function to create
            # the 4D causal mask exists, it should be present in the base model (XXXModel class) or in its decoder.
            base_model = getattr(self, self.base_model_prefix, self)
            decoder = base_model.get_decoder() if hasattr(base_model, "get_decoder") else None
            causal_mask_creation_function = getattr(
                base_model, "_prepare_4d_causal_attention_mask_with_cache_position", None
            )
            if causal_mask_creation_function is None and decoder is not None:  # it may be in the decoder
                causal_mask_creation_function = getattr(
                    decoder, "_prepare_4d_causal_attention_mask_with_cache_position", None
                )

            # If it's not defined, it means the model uses the new general mask API
            if causal_mask_creation_function is None:  # can't be found
                token_type_ids = model_inputs.get("token_type_ids")
                position_ids = model_inputs.get(position_ids_key)
                # Some models may overwrite the general one
                causal_mask_creation_function = getattr(self, "create_masks_for_generate", create_masks_for_generate)
                attention_mask_ = causal_mask_creation_function(
                    config=self.config,
                    # we only need batch size, seq_length and dtype here - we don't care about the values of the embeddings
                    input_embeds=torch.empty((batch_size, sequence_length), dtype=self.dtype),
                    attention_mask=attention_mask_,
                    cache_position=cache_position,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                    token_type_ids=token_type_ids,
                )
            else:
                attention_mask_ = causal_mask_creation_function(
                    attention_mask_,
                    sequence_length=sequence_length,
                    target_length=past_key_values.get_max_cache_shape(),
                    dtype=self.dtype,
                    cache_position=cache_position,
                    batch_size=batch_size,
                    config=self.config,
                    past_key_values=past_key_values,
                )
        if attention_mask_ is not None:
            if past_key_values is None or len(model_inputs['past_key_values'])==0:
                # print('use omni_attn_mask_naive')
                attention_mask = omni_attn_mask_naive(attention_mask_, [])
            elif self.SEG_START=='1':
                # print('use 10 full attn mask')
                attention_mask = fused_full_attn_mask(self.config.max_seg_nums+3, attention_mask_.shape[1]-1, attention_mask_)
            else:
                # print('use full attn mask')
                attention_mask = full_attn_mask(1, attention_mask_.shape[1], attention_mask_)
            # print(attention_mask.shape)
            model_inputs[attention_mask_key] = attention_mask

        if encoder_attention_mask is not None:
            model_inputs["attention_mask"] = encoder_attention_mask

        # 7. Forward ALL kwargs that are uninitialized (e.g. `use_cache`).
        for key, value in kwargs.items():
            if key not in model_inputs:
                model_inputs[key] = value

        # 8. Remove unexpected `generate` inputs (TODO @joao: fix trainer and examples)
        model_inputs.pop("labels", None)

        # Qwen3VL position_ids are prepared with rope_deltas
        if position_ids is None:
            # Calculate RoPE index once per generation in the pre-fill stage only.
            # When compiling, we can't check tensor values thus we check only input length
            # It is safe to assume that `length!=1` means we're in pre-fill because compiled
            # models currently cannot do asssisted decoding
            if model_inputs["cache_position"][0] == 0 or self.model.rope_deltas is None:
                vision_positions, rope_deltas = self.model.get_rope_index(
                    model_inputs.get("input_ids", None),
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    attention_mask=attention_mask,
                )
                self.model.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            elif "position_ids" in model_inputs:
                batch_size, seq_length = model_inputs["position_ids"].shape
                device = model_inputs["position_ids"].device
                position_ids = torch.arange(seq_length, device=device)
                position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
                delta = cache_position[0] + self.model.rope_deltas
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                vision_positions = position_ids + delta.expand_as(position_ids)

            # Concatenate "text + vision" positions into [4, bs, seq-len]
            text_positions = model_inputs["position_ids"][None, ...]
            model_inputs["position_ids"] = torch.cat([text_positions, vision_positions], dim=0)

        if cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        return model_inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> dict[str, Any]:
        seg_start = self.SEG_START
        if seg_start is not None:
            self.seg_output_embeddings.append(outputs['hidden_states'][-1][:,2:-1]) # except the start and end token 
            num_new_tokens = self.config.max_seg_nums

        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs=outputs,
            model_kwargs=model_kwargs,
            is_encoder_decoder=is_encoder_decoder,
            num_new_tokens=num_new_tokens
        )

        if self.SEG_START=='1':
            attention_mask = model_kwargs['attention_mask']
            model_kwargs['attention_mask'] = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], self.config.max_seg_nums+2))], dim=-1)
 
        return model_kwargs
