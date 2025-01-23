
import random
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from adapters import OutputParallelAdapterLayer, TaskEmbeddingController
from adapters.hypercomplex.layers import PHMLinear
from my_transformers.modeling_bart import (BartDecoder, BartEncoder,
                                           BartForConditionalGeneration,
                                           BartModel)
from prompt import PromptController
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import (BaseModelOutput, ModelOutput,
                                           Seq2SeqLMOutput, Seq2SeqModelOutput)
from transformers.models.bart.modeling_bart import (BartConfig,
                                                    shift_tokens_right)
from transformers.utils import logging

logger = logging.get_logger(__name__)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)

class CustomForward(nn.Module):
    def __init__(self, bert_module):
        super().__init__()
        self.bert_module = bert_module

    def forward(self, inputs):
        return self.bert_module(inputs_embeds=inputs).last_hidden_state


class VisualEmbedding(nn.Module):
    def __init__(self, config, obj_order_embedding):
        super().__init__()
        self.config = config
        feat_dim = config.feat_dim
        pos_dim = config.pos_dim
        # n_objs = config.n_objs
        n_images = config.n_images

        self.vis_use_transformer = config.vis_use_transformer

        # Object feature encoding

        feat_embedding = [nn.Linear(feat_dim, config.d_model)]

        if self.vis_use_transformer:
            bert_config = transformers.BertConfig(
                vocab_size=1,
                hidden_size=config.d_model,
                num_hidden_layers=2,
                num_attention_heads=12
            )

            transformer_model = transformers.BertModel(bert_config)

            feat_embedding.append(
                CustomForward(transformer_model)
            )

        # use custom layer norm
        if self.config.use_vis_layer_norm and self.config.individual_vis_layer_norm:
            feat_embedding.append(nn.LayerNorm(config.d_model))

        self.feat_embedding = nn.Sequential(*feat_embedding)

        absolute_vis_pos_embedding = [nn.Linear(pos_dim + 1, config.d_model)]

        # use custom layer norm
        if self.config.use_vis_layer_norm and self.config.individual_vis_layer_norm:
            absolute_vis_pos_embedding.append(nn.LayerNorm(config.d_model))
        self.absolute_vis_pos_embedding = nn.Sequential(*absolute_vis_pos_embedding)

        if self.config.use_vis_order_embedding:
            self.obj_order_embedding = obj_order_embedding
            self.img_order_embedding = nn.Embedding(n_images, config.d_model)

            self.default_obj_order_ids = self.config.default_obj_order_ids

        # use one layer norm
        if self.config.use_vis_layer_norm and not self.config.individual_vis_layer_norm:
            self.layer_norm = nn.LayerNorm(config.d_model)

    def get_area(self, pos):
        """
        Args
            pos: [B, N, 4]
                (x1, x2, y1, y2)
        Return
            area : [B, N]
        """
        # [B, N]
        height = pos[:, :, 3] - pos[:, :, 2]
        width = pos[:, :, 1] - pos[:, :, 0]
        area = height * width
        return area

    def forward(self, feats, pos, img_order_ids=None, obj_order_ids=None):
        """
        Args
            feats: [B, N, feat_dim]
            pos: [B, N, 4]
                (x1, x2, y1, y2)
        Return
            relative_vis_pos_embedding: [B, N, N, n_heads]
            absolute_vis_pos_embedding: # [B, N, d_model]
        """

        B, N, _ = feats.size()
        assert pos.size() == (B, N, 4)

        feat_embedding = self.feat_embedding(feats)

        device = feats.device

        area = self.get_area(pos).unsqueeze(2)  # [B, N, 1]
        pos = torch.cat([pos, area], dim=2)  # [B, N, 5]

        # [B, N, d_model]
        absolute_vis_pos_embedding = self.absolute_vis_pos_embedding(pos)

        if self.config.use_vis_order_embedding:
            if img_order_ids is None:
                img_order_ids = torch.zeros(N, dtype=torch.long, device=device)
                img_order_ids = img_order_ids.unsqueeze(0)  # .expand(B, -1)
            img_order_embedding = self.img_order_embedding(img_order_ids)

            if obj_order_ids is None:
                obj_order_ids = torch.arange(N, dtype=torch.long, device=device)
                obj_order_ids = obj_order_ids.unsqueeze(0)
            # print('raw obj_order_ids', obj_order_ids)
            obj_order_ids = self.obj_order_embedding.num_embeddings - obj_order_ids - 1
            # print('re-indexed obj_order_ids', obj_order_ids)
            obj_order_embedding = self.obj_order_embedding(obj_order_ids)

            vis_embedding = feat_embedding + absolute_vis_pos_embedding + \
                img_order_embedding + obj_order_embedding

        else:
            vis_embedding = feat_embedding + absolute_vis_pos_embedding

        if not self.config.individual_vis_layer_norm:
            if self.config.use_vis_layer_norm:
                vis_embedding = self.layer_norm(vis_embedding)

        return vis_embedding


class ExpandVisualEmbedding(nn.Module):
    def __init__(self, config, obj_order_embedding):
        super().__init__()
        self.config = config
        feat_dim = config.feat_dim
        pos_dim = config.pos_dim
        # n_objs = config.n_objs
        n_images = config.n_images

        n_image_tokens = config.n_image_tokens

        self.n_image_tokens = n_image_tokens

        # Object feature encoding
        feat_embedding = [nn.Linear(feat_dim, n_image_tokens * config.d_model)]

        # use custom layer norm
        if self.config.use_vis_layer_norm and self.config.individual_vis_layer_norm:
            feat_embedding.append(nn.LayerNorm(n_image_tokens * config.d_model))

        self.feat_embedding = nn.Sequential(*feat_embedding)

        absolute_vis_pos_embedding = [nn.Linear(pos_dim + 1, n_image_tokens * config.d_model)]

        # use custom layer norm
        if self.config.use_vis_layer_norm and self.config.individual_vis_layer_norm:
            absolute_vis_pos_embedding.append(nn.LayerNorm(n_image_tokens * config.d_model))
        self.absolute_vis_pos_embedding = nn.Sequential(*absolute_vis_pos_embedding)

        if self.config.use_vis_order_embedding:
            self.obj_order_embedding = nn.Embedding(config.n_boxes, n_image_tokens * config.d_model)
            self.img_order_embedding = nn.Embedding(n_images, n_image_tokens * config.d_model)

            self.default_obj_order_ids = self.config.default_obj_order_ids

        # use one layer norm
        if self.config.use_vis_layer_norm and not self.config.individual_vis_layer_norm:
            self.layer_norm = nn.LayerNorm(n_image_tokens * config.d_model)

    def get_area(self, pos):
        """
        Args
            pos: [B, N, 4]
                (x1, x2, y1, y2)
        Return
            area : [B, N]
        """
        # [B, N]
        height = pos[:, :, 3] - pos[:, :, 2]
        width = pos[:, :, 1] - pos[:, :, 0]
        area = height * width
        return area

    def forward(self, feats, pos, img_order_ids=None, obj_order_ids=None):
        """
        Args
            feats: [B, N, feat_dim]
            pos: [B, N, 4]
                (x1, x2, y1, y2)
        Return
            relative_vis_pos_embedding: [B, N, N, n_heads]
            absolute_vis_pos_embedding: # [B, N, d_model]
        """

        B, N, _ = feats.size()
        assert pos.size() == (B, N, 4)

        feat_embedding = self.feat_embedding(feats)

        device = feats.device
        dtype = feats.dtype

        area = self.get_area(pos).unsqueeze(2)  # [B, N, 1]
        pos = torch.cat([pos, area], dim=2)  # [B, N, 5]

        # [B, N, d_model]
        absolute_vis_pos_embedding = self.absolute_vis_pos_embedding(pos)

        if self.config.use_vis_order_embedding:
            if img_order_ids is None:
                img_order_ids = torch.zeros(N, dtype=torch.long, device=device)
                img_order_ids = img_order_ids.unsqueeze(0)  # .expand(B, -1)
            img_order_embedding = self.img_order_embedding(img_order_ids)

            if obj_order_ids is None:
                obj_order_ids = torch.arange(N, dtype=torch.long, device=device)
                obj_order_ids = obj_order_ids.unsqueeze(0)
            # print('re-indexed obj_order_ids', obj_order_ids)
            obj_order_embedding = self.obj_order_embedding(obj_order_ids)

            vis_embedding = feat_embedding + absolute_vis_pos_embedding + \
                img_order_embedding + obj_order_embedding

        else:
            vis_embedding = feat_embedding + absolute_vis_pos_embedding

        if not self.config.individual_vis_layer_norm:
            if self.config.use_vis_layer_norm:
                vis_embedding = self.layer_norm(vis_embedding)

        vis_embedding = vis_embedding.reshape(B, self.n_image_tokens, -1)

        return vis_embedding


class Downsample(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        """
        output size: list of 1-D size, such as (6, 6), (16, 16)
        """
        self.output_size = output_size
        self.pool = nn.AdaptiveMaxPool2d(output_size)

    def downsample_inputs(self, inputs):
        B, L, dim = inputs.shape

        inputs = inputs.permute(0, 2, 1) # (2B, dim, L/2)

        # restriction: L**0.5 must to be integer
        sqrt_L = int(L ** 0.5)

        inputs = inputs.reshape(B, dim, sqrt_L, sqrt_L)

        inputs = self.pool(inputs) # (B, dim, self.output_size[0], self.output_size[1])
        inputs = inputs.reshape(B, dim, -1)

        inputs = inputs.permute(0, 2, 1) # (2B, self.output_size[0]**2, dim)

        return inputs

    def forward(self, inputs_tuple):
        # inputs (B, L, dim)

        if len(inputs_tuple) == 4: # (NLVR)
            inputs, boxes, img_order_ids, obj_order_ids = inputs_tuple

            inputs = torch.cat(torch.chunk(inputs, 2, 1), 0) # (2B, L/2, dim)
            inputs = self.downsample_inputs(inputs)
            inputs = torch.cat(torch.chunk(inputs, 2, 0), 1) # (B, L, dim)

            boxes = torch.cat(torch.chunk(boxes, 2, 1), 0)
            boxes = boxes[:, :inputs.shape[1]//2]
            boxes = torch.cat(torch.chunk(boxes, 2, 0), 1)

            img_order_ids = torch.cat(torch.chunk(img_order_ids, 2, 1), 0)
            img_order_ids = img_order_ids[:, :inputs.shape[1]//2]
            img_order_ids = torch.cat(torch.chunk(img_order_ids, 2, 0), 1)

            obj_order_ids = torch.cat(torch.chunk(obj_order_ids, 2, 1), 0)
            obj_order_ids = obj_order_ids[:, :inputs.shape[1]//2]
            obj_order_ids = torch.cat(torch.chunk(obj_order_ids, 2, 0), 1)

            outputs_tuple = (inputs, boxes, img_order_ids, obj_order_ids)
        else:
            inputs, boxes = inputs_tuple
            
            inputs = self.downsample_inputs(inputs)
            boxes = boxes[:, :inputs.shape[1]] # Get the first few data because the element are all zeros

            outputs_tuple = (inputs, boxes)

        return outputs_tuple


class OneDDownsample(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        """
        output size: list of 1-D size, such as (6, 6), (16, 16)
        """
        self.output_size = output_size
        self.pool = nn.AdaptiveMaxPool1d(output_size)

    def downsample_inputs(self, inputs):
        B, L, dim = inputs.shape

        inputs = inputs.permute(0, 2, 1) # (2B, dim, L/2)

        inputs = self.pool(inputs) # (B, dim, self.output_size[0], self.output_size[1])
        inputs = inputs.reshape(B, dim, -1)

        inputs = inputs.permute(0, 2, 1) # (2B, self.output_size[0]**2, dim)

        return inputs

    def forward(self, inputs_tuple):
        # inputs (B, L, dim)

        if len(inputs_tuple) == 4: # (NLVR)
            inputs, boxes, img_order_ids, obj_order_ids = inputs_tuple

            inputs = torch.cat(torch.chunk(inputs, 2, 1), 0) # (2B, L/2, dim)
            inputs = self.downsample_inputs(inputs)
            inputs = torch.cat(torch.chunk(inputs, 2, 0), 1) # (B, L, dim)

            boxes = torch.cat(torch.chunk(boxes, 2, 1), 0)
            boxes = boxes[:, :inputs.shape[1]//2]
            boxes = torch.cat(torch.chunk(boxes, 2, 0), 1)

            img_order_ids = torch.cat(torch.chunk(img_order_ids, 2, 1), 0)
            img_order_ids = img_order_ids[:, :inputs.shape[1]//2]
            img_order_ids = torch.cat(torch.chunk(img_order_ids, 2, 0), 1)

            obj_order_ids = torch.cat(torch.chunk(obj_order_ids, 2, 1), 0)
            obj_order_ids = obj_order_ids[:, :inputs.shape[1]//2]
            obj_order_ids = torch.cat(torch.chunk(obj_order_ids, 2, 0), 1)

            outputs_tuple = (inputs, boxes, img_order_ids, obj_order_ids)
        else:
            inputs, boxes = inputs_tuple
            
            inputs = self.downsample_inputs(inputs)
            boxes = boxes[:, :inputs.shape[1]] # Get the first few data because the element are all zeros

            outputs_tuple = (inputs, boxes)

        return outputs_tuple


class SparseSample(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        """
        output size: list of 1-D size, such as (6, 6), (16, 16)
        """
        self.output_size = output_size

    def forward(self, inputs):
        if self.training:
            B, L, _ = inputs.shape

            x = torch.rand(B, L)

            indices = torch.argsort(torch.rand(*x.shape), dim=-1)

            indices = indices[:, :self.output_size]

            indices = torch.sort(indices)[0]

            return inputs[torch.arange(B).unsqueeze(1), indices]
        else:
            return inputs


class JointEncoder(BartEncoder):
    """
    BartEncoder + visual embedding
    """
    def __init__(self, config, embed_tokens=None, task_embed=None):
        super().__init__(config, embed_tokens, task_embed)

        self.config = config

        self.visual_embedding = VisualEmbedding(config, self.embed_tokens)

        if config.expand_vis_embedding: # This is not use right now, this is for Frozen type visual embeddings
            self.visual_embedding = ExpandVisualEmbedding(config, self.embed_tokens)

        self.downsample = None
        self.sparse_sample = None
        if config.oneddownsample:
            self.downsample = OneDDownsample(config.n_boxes)
        elif config.downsample:
            sqrt_size = int(config.n_boxes ** 0.5)
            output_size = (sqrt_size, sqrt_size)
            self.downsample = Downsample(output_size)
        elif config.sparse_sample:
            self.sparse_sample = SparseSample(config.n_boxes)

        if config.encoder_prompt_config:
            self.prompt_modules = PromptController(config.encoder_prompt_config)
        else:
            self.prompt_modules = None

        self.init_weights()

    def get_prompt(self, bsz, device):
        input_tokens = self.prefix_tokens.unsqueeze(0).expand(bsz, -1).to(device)
        return self.prefix_embedding(input_tokens)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,

        vis_inputs=None,
        vis_attention_mask=None,

        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        past_key_values=None,
        return_dict=None,
        task=None
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(input_shape)

        inputs_embeds = inputs_embeds + embed_pos

        if self.prompt_modules is not None:
            prefix_embeds = self.prompt_modules(inputs_embeds.shape[0], inputs_embeds.device, task)
            inputs_embeds = torch.cat([prefix_embeds, inputs_embeds], dim=1)

        B, L = inputs_embeds.size()[:-1]

        if vis_inputs is not None:

            if self.downsample is not None:
                vis_inputs = self.downsample(vis_inputs)

            vis_feats = vis_inputs[0]
            boxes = vis_inputs[1]
            img_order_ids = None
            obj_order_ids = None
            if len(vis_inputs) >= 3:
                img_order_ids = vis_inputs[2]
            if len(vis_inputs) == 4:
                obj_order_ids = vis_inputs[3]

            vis_embeds = self.visual_embedding(vis_feats, boxes, img_order_ids, obj_order_ids)

            if self.sparse_sample is not None:
                vis_embeds = self.sparse_sample(vis_embeds)

            V_L = vis_embeds.size(1)

            if self.config.share_vis_lang_layer_norm:
                inputs_embeds = torch.cat([inputs_embeds, vis_embeds], dim=1)

                inputs_embeds = self.layernorm_embedding(inputs_embeds)
            else:
                inputs_embeds = self.layernorm_embedding(inputs_embeds)
                inputs_embeds = torch.cat([inputs_embeds, vis_embeds], dim=1)

            if vis_attention_mask is None:
                vis_attention_mask = torch.ones(B, V_L, dtype=inputs_embeds.dtype, device=inputs_embeds.device)

        else:
            inputs_embeds = self.layernorm_embedding(inputs_embeds)

        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id).to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)

        hidden_states = F.dropout(inputs_embeds, p=self.dropout, training=self.training)

        # print('attention_mask, ', attention_mask.size())
        # print('vis_attention_mask, ', vis_attention_mask.size())

        if self.prompt_modules is not None:
            prefix_attention_mask = torch.ones(
                B, prefix_embeds.shape[1], dtype=inputs_embeds.dtype, device=inputs_embeds.device
            )

            attention_mask = torch.cat([prefix_attention_mask, attention_mask], dim=1)

        if vis_attention_mask is not None:
            attention_mask = torch.cat([attention_mask, vis_attention_mask], dim=1)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        task_embedding = None
        if task is not None and self.task_embed is not None:
            task_embedding = self.task_embed(task)

        # print('ext_attention_mask, ', attention_mask.size())
        # print('attention_mask')
        # print(attention_mask.size())
        # print(attention_mask)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)

            block_adapters = None
            if self.adapter_layers_hyper_net:
                block_adapters = self.adapter_layers_hyper_net(task_embedding, idx)

            # for prefix
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if getattr(self.config, "gradient_checkpointing", False):

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        past_key_value,
                        block_adapters,
                        task,
                    )
                else:
                    layer_outputs = encoder_layer(hidden_states, attention_mask, past_key_value, block_adapters, task=task, output_attentions=output_attentions)

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )



class VLBartModel(BartModel):
    def __init__(self, config: BartConfig):
        super(BartModel, self).__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        if config.use_hyperformer:
            self.shared_task_embed = TaskEmbeddingController(config.adapter_config)
        else:
            self.shared_task_embed = None
        
        #----- Modified-----#
        # self.encoder = BartEncoder(config, self.shared)

        self.encoder = JointEncoder(config, self.shared, self.shared_task_embed)
        #-------------------#
        self.decoder = BartDecoder(config, self.shared, self.shared_task_embed)

        self.config = config

        if config.decoder_prompt_config:
            self.prompt_modules = PromptController(config.decoder_prompt_config)
        else:
            self.prompt_modules = None

        self.init_weights()

    def get_prompt(self, bsz, device):
        input_tokens = self.prefix_tokens.unsqueeze(0).expand(bsz, -1).to(device) # (B, L)
        prefix_prompt = self.prefix_embedding(input_tokens) # (B, L, d_model)

        temp_results = self.decoder(inputs_embeds=prefix_prompt, use_cache=True, return_dict=True)

        past_key_values = temp_results.past_key_values

        return past_key_values

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

        if not self.config.expand_vis_embedding:
            self.encoder.visual_embedding.obj_order_embedding = self.shared

    def forward(
        self,
        input_ids=None,
        attention_mask=None,

        vis_inputs=None,
        vis_attention_mask=None,

        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task=None,
        **kwargs,
    ):

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided

        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,

                vis_inputs=vis_inputs,
                vis_attention_mask=vis_attention_mask,

                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                task=task,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=False
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id).to(dtype=torch.float, device=input_ids.device)
        
        if self.config.encoder_prompt_config is not None and self.config.encoder_prompt_config.prompt_len > 0:
            prefix_attention_mask = torch.ones(
                attention_mask.shape[0], 
                self.config.encoder_prompt_config.prompt_len, 
                dtype=attention_mask.dtype, 
                device=attention_mask.device,
            )

            attention_mask = torch.cat([prefix_attention_mask, attention_mask], dim=1)
        
        if vis_attention_mask is None:
            B, L = attention_mask.size()
            V_L = encoder_outputs[0].size(1) - L
            vis_attention_mask = attention_mask.new_ones(B, V_L)

        encoder_attention_mask = torch.cat([attention_mask, vis_attention_mask], dim=1)

        if self.prompt_modules is not None and past_key_values is None:
            prefix_embeds = self.prompt_modules(B, attention_mask.device, task)

            past_key_values = self.decoder(inputs_embeds=prefix_embeds, use_cache=True, return_dict=True).past_key_values

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            # encoder_attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            task=task,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class VLBart(BartForConditionalGeneration):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [
        r"final_logits_bias",
        r"encoder\.version",
        r"decoder\.version",
        r"lm_head\.weight",
    ]

    def __init__(self, config: BartConfig):
        super(BartForConditionalGeneration, self).__init__(config)
        self.model = VLBartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        self.output_adapter = None
        if config.use_lm_head_adapter:
            self.output_adapter = OutputParallelAdapterLayer(config, self.model.shared.num_embeddings)

        adapter_config = config.adapter_config

        if adapter_config is not None:
            if getattr(adapter_config, "train_task_adapters", None) and getattr(adapter_config, "hypercomplex_adapters", None):
                if adapter_config.shared_phm_rule:
                    phm_dim = adapter_config.hypercomplex_division
                    self.factorized_phm_rule = adapter_config.factorized_phm_rule
                    if self.factorized_phm_rule:
                        self.phm_rule_left = nn.Parameter(torch.FloatTensor(phm_dim, phm_dim, 1),
                            requires_grad=adapter_config.learn_phm)
                        self.phm_rule_right = nn.Parameter(torch.FloatTensor(phm_dim, 1, phm_dim),
                            requires_grad=adapter_config.learn_phm)
                        if adapter_config.phm_c_init == "normal":
                            self.phm_rule_left.data.normal_(mean=0, std=adapter_config.phm_init_range)
                            self.phm_rule_right.data.normal_(mean=0, std=adapter_config.phm_init_range)
                        elif adapter_config.phm_c_init == "uniform":
                            self.phm_rule_left.data.uniform_(-1, 1)
                            self.phm_rule_right.data.uniform_(-1, 1)
                        else:
                            raise NotImplementedError
                    else:
                        self.phm_rule = nn.Parameter(torch.FloatTensor(phm_dim, phm_dim, phm_dim),\
                            requires_grad=adapter_config.learn_phm)
                        if adapter_config.phm_c_init == "normal":
                            self.phm_rule.data.normal_(mean=0, std=adapter_config.phm_init_range)
                        elif adapter_config.phm_c_init == "uniform":
                            self.phm_rule.data.uniform_(-1, 1)
                        else:
                            raise NotImplementedError 
                    self.set_phm_rule()

        self.init_weights()

    def set_phm_rule(self):
        def set_phm_rule(module):
            # TODO: we need to check there is one of these, and this is activated.
            for name, sub_module in module.named_modules():
                if isinstance(sub_module, PHMLinear):
                    if self.factorized_phm_rule:
                        sub_module.set_phm_rule(phm_rule_right=self.phm_rule_right, 
                                                phm_rule_left=self.phm_rule_left)
                    else:
                        sub_module.set_phm_rule(phm_rule=self.phm_rule)
        set_phm_rule(self.model)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,

        vis_inputs=None,
        vis_attention_mask=None,

        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task=None,

        reduce_loss=False,

        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,

            vis_inputs=vis_inputs,
            vis_attention_mask=vis_attention_mask,

            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            task=task,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        if self.output_adapter is not None:
            lm_logits = lm_logits + self.output_adapter(outputs[0])

        masked_lm_loss = None
        if labels is not None:
            # loss_fct = CrossEntropyLoss()
            if reduce_loss:
                loss_fct = CrossEntropyLoss(ignore_index=-100)
            else:
                loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def vis_forward(self, batch, device):
        if hasattr(self, "vis_encoder"):
            # self.vis_encoder.eval() # freeze the batchnorm statistics
            images = batch["images"].to(device)
            # print("send raw images to vis_encoder")
            if self.config.vis_pooling_output:
                _, vis_feats = self.vis_encoder(images)
            else:
                vis_feats, _ = self.vis_encoder(images)
            # vis_feats: (B, dim, L ** 0.5, L ** 0.5)
            B, L, D = vis_feats.shape
            vis_pos = torch.zeros(B, L, 4, dtype=vis_feats.dtype)

            batch["vis_feats"] = vis_feats
            batch["boxes"] = vis_pos

        return batch

    def prepare_inputs_for_generation(
        self, decoder_input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        output = {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

        if "vis_attention_mask" in kwargs:
            output["vis_attention_mask"] = kwargs['vis_attention_mask']

        if "task" in kwargs:
            output["task"] = kwargs["task"]

        return output

    @staticmethod
    def _expand_inputs_for_generation(
        input_ids: torch.LongTensor,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        attention_mask: torch.LongTensor = None,
        encoder_outputs: ModelOutput = None,
        **model_kwargs
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1,
                                                                expand_size).view(-1).to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(
                0, expanded_return_idx)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(
                0, expanded_return_idx)

        if model_kwargs.get("vis_attention_mask", None) is not None:
            model_kwargs['vis_attention_mask'] = model_kwargs['vis_attention_mask'].index_select(
                0, expanded_return_idx)

        if is_encoder_decoder:
            assert encoder_outputs is not None
            encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
                0, expanded_return_idx
            )
            model_kwargs["encoder_outputs"] = encoder_outputs

        return input_ids, model_kwargs
