from transformers import AutoConfig
from .sam3 import Sam3Config, Sam3Model
import torch.nn as nn
class SegmentationDecoder(nn.Module):
    def __init__(self, config):
        super(SegmentationDecoder, self).__init__()
        self.config = config
        if config.seg_encoder=='sam3' and config.seg_decoder=='sam3':
            # config = Sam3Config.from_pretrained(config.mask_decoder_model)
            config = AutoConfig.from_pretrained(self.config.mask_decoder_model)
            config.detector_config.detr_decoder_config.num_queries = self.config.max_seg_nums
        
            self.model = Sam3Model(config)
            # self.mask_encoder = self.model.vision_encoder
            # self.mask_decoder = self.model
        else:
            raise NotImplementedError

    def load_model(self, config):
        original_config = AutoConfig.from_pretrained(self.config.mask_decoder_model)
        original_config.detector_config.detr_decoder_config.num_queries = config.max_seg_nums
        self.model = Sam3Model.from_pretrained(self.config.mask_decoder_model, config=original_config, ignore_mismatched_sizes=True)

    def encoder(self, pixel_values):
        vision_outputs = self.model.vision_encoder(pixel_values)
        return vision_outputs


    def decoder(self, vision_outputs, text_embeds, text_attn_mask, query_embed):
        mask_outputs = self.model(
            vision_embeds = vision_outputs,
            attention_mask=text_attn_mask,
            text_embeds = text_embeds,
            query_embed = query_embed,
        )
        return mask_outputs