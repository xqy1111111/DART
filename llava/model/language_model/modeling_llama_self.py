import torch
from typing import Tuple, Callable
from transformers import AutoConfig
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaDecoderLayer, LlamaModel, _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa, Cache, DynamicCache
from transformers.models.llama import LlamaConfig
from transformers.modeling_outputs import BaseModelOutputWithPast
from typing import List, Optional, Tuple, Union
import numpy as np
import random
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import pdb
import json
import time
from einops import rearrange
import math

def get_retained_image_token(config: LlamaConfig, last_layer_state: torch.Tensor, K_states: torch.Tensor) -> torch.Tensor:

    DART_config = config.DART_config
    K = DART_config['K']  # pruned layer
    image_token_start_index = DART_config['image_token_start_index']
    image_token_length = DART_config['image_token_length']
    MAX_NUM_TRUNCTION = DART_config['max_num_trunction']

    pivot_image_token = DART_config['pivot_image_token']  # pivot token
    pivot_text_token = DART_config['pivot_text_token']  # pivot token
    TOKEN_TOPK = int(MAX_NUM_TRUNCTION // (pivot_image_token + pivot_text_token))

    device = last_layer_state.device

    if config.text_length is not None:
        text_length = config.text_length
        image_token_length = K_states.shape[2] - text_length  # layer_outputs[-2] == K_states

        retain_token_num_for_llava_next = DART_config['retain_token_num_for_llava_next']
        retain_token_num_for_llava_next = min(retain_token_num_for_llava_next, image_token_length - pivot_image_token)
        TOKEN_TOPK = int(retain_token_num_for_llava_next // (pivot_image_token + pivot_text_token))

    K_states = K_states.permute(0, 2, 1, 3).reshape(K_states.shape[0], K_states.shape[1], -1)  

    k_states_image_token = K_states[0][image_token_start_index:image_token_start_index + image_token_length, :]
    k_states_query_token = K_states[0][image_token_start_index + image_token_length:, :]

    k_states_image_token_L1_norm = torch.norm(k_states_image_token, p=1, dim=-1)
    k_states_query_token_L1_norm = torch.norm(k_states_query_token, p=1, dim=-1)

    image_indices = (k_states_image_token_L1_norm.topk(pivot_image_token).indices + image_token_start_index).tolist()
    query_indices = (k_states_query_token_L1_norm.topk(pivot_text_token).indices + image_token_start_index + image_token_length).tolist()
    indices_set = set(image_indices + query_indices)

    tmp_set = indices_set.copy()

    valid_indices = set(range(image_token_start_index, image_token_start_index + image_token_length))
    valid_indices.difference_update(image_indices)

    for item in list(tmp_set):
        valid_indices_list = list(valid_indices)
        valid_vectors = last_layer_state[0][valid_indices_list, :]

        cos_sim = -torch.nn.functional.cosine_similarity(
            last_layer_state[0][item, :], valid_vectors, dim=-1
        )

        top_k_indices = cos_sim.topk(TOKEN_TOPK).indices
        top_k_real_indices = [valid_indices_list[i] for i in top_k_indices.tolist()]

        indices_set.update(top_k_real_indices)
        valid_indices.difference_update(top_k_real_indices)

    indices_set.difference_update(query_indices)

    # keep index
    retained_image_tokens_index = torch.tensor(list(indices_set), device=device)

    return retained_image_tokens_index


# HACK:k_norm and topk(, largest=True)
class DART_K_norm(LlamaModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        self.last_attention = None
        super().__init__(config)
        self.config = config

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                print(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)


        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                DART_config = self.config.DART_config
                if DART_config is not None:
                    K = DART_config['K']  # pruned layer
                    image_token_start_index = DART_config['image_token_start_index']
                    image_token_length = DART_config['image_token_length']
                    
                    if decoder_layer.self_attn.layer_idx == K and seq_length > 1:
                        device = hidden_states.device

                        last_layer_state = layer_outputs[0]
                        last_layer_state = self.norm(last_layer_state)
                        K_states = layer_outputs[-2]  # HACK: get K_states

                        # keep index
                        retained_image_tokens_index = get_retained_image_token(self.config, last_layer_state, K_states)

                        keep_indexs = torch.cat((torch.arange(image_token_start_index,device=device), retained_image_tokens_index, torch.arange(image_token_start_index+image_token_length,seq_length,device=device)))
                        # sort index
                        keep_indexs = keep_indexs.sort().values

                        hidden_states = hidden_states[:,keep_indexs,:]
                        if attention_mask is not None:
                            attention_mask = attention_mask[:,:,:hidden_states.shape[1],:hidden_states.shape[1]]
                        position_ids = keep_indexs.unsqueeze(0)


                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


