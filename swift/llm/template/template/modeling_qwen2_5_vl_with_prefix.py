"""
Qwen2.5-VL model with text prefix support for vision encoder.
This extends the base Qwen2.5-VL implementation to support using text instructions
as prefix for the vision encoder.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLModel,
    Qwen2_5_VLPreTrainedModel,
    BaseModelOutputWithPast,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerationMixin

@dataclass
class Qwen2_5_VLWithPrefixConfig(Qwen2_5_VLConfig):
    """Configuration class for Qwen2.5-VL with text prefix support."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prefix_hidden_size = kwargs.pop("prefix_hidden_size", self.text_config.hidden_size)
        self.prefix_cross_attention_layers = kwargs.pop("prefix_cross_attention_layers", 2)
        self.prefix_max_length = kwargs.pop("prefix_max_length", 512)
        
        # Add auto mapping for our model
        self.auto_map = {
            "AutoModelForCausalLM": "modeling_qwen2_5_vl_with_prefix.Qwen2_5_VLWithPrefixForConditionalGeneration",
            "AutoModel": "modeling_qwen2_5_vl_with_prefix.Qwen2_5_VLWithPrefixModel",
        }
        self.architectures = ["Qwen2_5_VLWithPrefixForConditionalGeneration"]

class Qwen2_5_VLWithPrefixModel(Qwen2_5_VLModel):
    """Qwen2.5-VL model with text prefix support for vision encoder."""
    
    config_class = Qwen2_5_VLWithPrefixConfig
    
    def __init__(self, config: Qwen2_5_VLWithPrefixConfig):
        super().__init__(config)
        
        # Text prefix encoder
        self.prefix_embeddings = nn.Embedding(config.text_config.vocab_size, config.prefix_hidden_size)
        self.prefix_encoder = nn.ModuleList([
            nn.MultiheadAttention(config.prefix_hidden_size, 8, batch_first=True)
            for _ in range(config.prefix_cross_attention_layers)
        ])
        
        # Project prefix embeddings to vision dimension
        self.prefix_proj = nn.Linear(config.prefix_hidden_size, config.vision_config.hidden_size)
        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[dict] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        text_prefix_ids: Optional[torch.LongTensor] = None,
        text_prefix_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """
        Forward pass with text prefix support.
        
        Args:
            text_prefix_ids: Text prefix token ids [batch_size, prefix_length]
            text_prefix_attention_mask: Attention mask for prefix tokens
            **kwargs: Additional arguments passed to parent class
            
        Returns:
            Model outputs with text prefix conditioning
        """
        # Process text prefix if provided
        if text_prefix_ids is not None:
            # Embed prefix tokens
            prefix_embeds = self.prefix_embeddings(text_prefix_ids)
            
            # Apply self-attention layers
            for layer in self.prefix_encoder:
                prefix_embeds = layer(
                    prefix_embeds,
                    prefix_embeds,
                    prefix_embeds,
                    key_padding_mask=~text_prefix_attention_mask.bool(),
                    need_weights=False,
                )[0]
            
            # Project to vision dimension
            prefix_embeds = self.prefix_proj(prefix_embeds)
            
            # Condition vision encoder with prefix
            if pixel_values is not None:
                vision_embeds = self.visual(pixel_values)
                vision_embeds = vision_embeds + prefix_embeds.mean(dim=1, keepdim=True)
                pixel_values = self.visual.proj(vision_embeds)
        
        # Forward through base model
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask, 
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pixel_values=pixel_values,
            **kwargs,
        )

class Qwen2_5_VLWithPrefixForConditionalGeneration(Qwen2_5_VLPreTrainedModel, GenerationMixin):
    """Qwen2.5-VL model with text prefix support for conditional generation."""
    
    config_class = Qwen2_5_VLWithPrefixConfig
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Qwen2_5_VLWithPrefixConfig):
        super().__init__(config)
        self.model = Qwen2_5_VLWithPrefixModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[dict] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        text_prefix_ids: Optional[torch.LongTensor] = None,
        text_prefix_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        """
        Prepare inputs for generation.
        
        Args:
            input_ids: Input token ids
            past_key_values: Past key values for attention
            attention_mask: Attention mask
            position_ids: Position ids
            pixel_values: Image pixel values
            text_prefix_ids: Text prefix token ids
            text_prefix_attention_mask: Attention mask for prefix tokens
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of model inputs
        """
        if past_key_values is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if position_ids is not None:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # Create position IDs if needed
        if position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values is not None:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "pixel_values": pixel_values,
            "text_prefix_ids": text_prefix_ids,
            "text_prefix_attention_mask": text_prefix_attention_mask,
            "use_cache": kwargs.get("use_cache", True),
        }

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[dict] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        text_prefix_ids: Optional[torch.LongTensor] = None,
        text_prefix_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass for conditional generation with text prefix support.
        
        Args:
            text_prefix_ids: Text prefix token ids [batch_size, prefix_length]
            text_prefix_attention_mask: Attention mask for prefix tokens
            **kwargs: Additional arguments passed to parent class
            
        Returns:
            Model outputs for conditional generation
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pixel_values=pixel_values,
            text_prefix_ids=text_prefix_ids,
            text_prefix_attention_mask=text_prefix_attention_mask,
            **kwargs,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        ) 