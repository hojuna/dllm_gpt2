import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from typing import Optional, Tuple, Union
from dataclasses import dataclass
from transformers.models.gpt2.modeling_gpt2 import (
    GPT2PreTrainedModel, 
    GPT2Attention, 
    GPT2Block, 
    GPT2Model,
    GPT2LMHeadModel,
    GPT2Config
)
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions, BaseModelOutputWithPastAndCrossAttentions
from transformers.utils import logging
from transformers.pytorch_utils import Conv1D

logger = logging.get_logger(__name__)

def block_diff_mask(b, h, q_idx, kv_idx, block_size=None, n=None):
    """
    Constructs the specialized block diffusion attention mask for training.
    """
    # Indicate whether token belongs to xt or x0
    x0_flag_q = (q_idx >= n)
    x0_flag_kv = (kv_idx >= n)

    # Compute block indices
    block_q = torch.where(x0_flag_q == 1,
                        (q_idx - n) // block_size,
                        q_idx // block_size)
    block_kv = torch.where(x0_flag_kv == 1,
                        (kv_idx - n) // block_size,
                        kv_idx // block_size)

    # **1. Block Diagonal Mask (M_BD) **
    block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)

    # **2. Offset Block-Causal Mask (M_OBC) **
    offset_block_causal = (
    (block_q > block_kv)
    & (x0_flag_kv == 1)
    & (x0_flag_q == 0)
    )

    # **3. Block-Causal Mask (M_BC) **
    block_causal = (block_q >= block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 1)

    # **4. Combine Masks **
    return block_diagonal | offset_block_causal | block_causal

def eval_block_diff_mask(q_idx, kv_idx, block_size=None):
    # Compute block indices
    block_q = q_idx // block_size
    block_kv = kv_idx // block_size

    return block_q >= block_kv

class GPT2dLLMAttention(GPT2Attention):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__(config, is_cross_attention, layer_idx)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
        **kwargs
    ):
        if layer_past is None and "past_key_value" in kwargs:
            layer_past = kwargs["past_key_value"]

        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        new_shape = query.size()[:-1] + (self.num_heads, self.head_dim)
        query = query.view(*new_shape).permute(0, 2, 1, 3)
        key = key.view(*new_shape).permute(0, 2, 1, 3)
        value = value.view(*new_shape).permute(0, 2, 1, 3)

        if layer_past is not None and len(layer_past) == 2:
            past_key, past_value = layer_past

            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        elif layer_past is not None:
            # 이 경로가 있나?
            pass

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        else:
            attn_weights = torch.matmul(query, key.transpose(-1, -2))

            if self.scale_attn_weights:
                attn_weights = attn_weights / (value.size(-1) ** 0.5)

            if self.scale_attn_by_inverse_layer_idx:
                attn_weights = attn_weights / float(self.layer_idx + 1)

            if not self.is_cross_attention:
                if attention_mask is not None:
                    if attention_mask.dtype == torch.bool:
                         mask_value = torch.finfo(attn_weights.dtype).min
                         attn_weights = torch.where(attention_mask, attn_weights, mask_value)
                    else:
                        attn_weights = attn_weights + attention_mask
            
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)

            if head_mask is not None:
                attn_weights = attn_weights * head_mask

            attn_output = torch.matmul(attn_weights, value)

        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        new_shape = attn_output.size()[:-2] + (self.embed_dim,)
        attn_output = attn_output.view(*new_shape)
        
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs

class GPT2dLLMBlock(GPT2Block):
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx)
        self.attn = GPT2dLLMAttention(config, layer_idx=layer_idx)

class GPT2dLLMModel(GPT2Model):
    def __init__(self, config):
        super().__init__(config)
        self.h = nn.ModuleList([GPT2dLLMBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.post_init()
        self.bd_size = getattr(config, "bd_size", 32) # Default block size

    def gen_mask(self, seqlen, block_size, B, H):
        q_indices = torch.arange(seqlen * 2).unsqueeze(1) 
        q_len = seqlen * 2
        kv_len = seqlen * 2
        
        q_idx = torch.arange(q_len).unsqueeze(1)
        kv_idx = torch.arange(kv_len).unsqueeze(0)
        
        mask = block_diff_mask(B, H, q_idx, kv_idx, block_size=block_size, n=seqlen)
        
        mask = mask.unsqueeze(0).unsqueeze(0)
        return mask

    def eval_mask(self, seqlen, block_size, cache_seq_len):
        q_indices = torch.arange(seqlen) + cache_seq_len
        k_indices = torch.arange(seqlen + cache_seq_len)
        
        mask = eval_block_diff_mask(
            q_idx=q_indices[:, None], 
            kv_idx=k_indices[None, :], 
            block_size=block_size
        )
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None, # Added for dLLM mask generation
        **kwargs
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        
        if labels is not None: 
             seq_len = input_ids.shape[1] // 2
             custom_mask = self.gen_mask(seq_len, self.bd_size, input_ids.shape[0], self.config.num_attention_heads)
             custom_mask = custom_mask.to(input_ids.device)
             
             attention_mask = custom_mask
        else:
            if attention_mask is None and input_ids is not None:
                 # Simple eval mask logic
                 pass 

        return super().forward(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

class GPT2dLLMLMHeadModel(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2dLLMModel(config)
        self.post_init()
        self.bd_size = getattr(config, "bd_size", 32)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        mask_id: Optional[int] = 50256, # Default to EOS/BOS if not specified, but should be specific MASK token
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        
        if labels is not None:
            original_labels = labels.clone()
            original_input_ids = input_ids.clone()

            noisy_input_ids = input_ids.clone()
            
            # Ensure length is divisible by bd_size
            b, l = input_ids.shape
            if l % self.bd_size != 0:
                new_l = (l // self.bd_size) * self.bd_size
                input_ids = input_ids[:, :new_l]
                labels = labels[:, :new_l]
                noisy_input_ids = noisy_input_ids[:, :new_l]
                original_input_ids = original_input_ids[:, :new_l]
                original_labels = original_labels[:, :new_l]
                l = new_l

            # 1. Primary Masking
            t = torch.rand((b,), device=input_ids.device)
            eps = 1e-3
            p_mask = (1 - eps) * t + eps
            p_mask = p_mask[:, None].repeat(1, l)
            
            mask_indices = torch.rand((b, l), device=input_ids.device) < p_mask
            x_t = torch.where(mask_indices, mask_id, input_ids)
            
            valid_mask = (labels != -100)
            noisy_input_ids[valid_mask] = x_t[valid_mask]
            
            is_masked = (noisy_input_ids == mask_id)
            labels[~is_masked] = -100
            
            # [Noisy, Clean]
            input_ids_primary = torch.cat([noisy_input_ids, original_input_ids], dim=1)
            
            # 2. Complementary Masking (Train on the inverse mask as well)
            complementary_noisy_input_ids = original_input_ids.clone()
            complementary_labels = original_labels.clone()
            
            complementary_mask_indices = ~mask_indices
            complementary_x_t = torch.where(complementary_mask_indices, mask_id, original_input_ids)
            
            complementary_noisy_input_ids[original_labels != -100] = complementary_x_t[original_labels != -100]
            
            complementary_is_masked = (complementary_noisy_input_ids == mask_id)
            complementary_labels[~complementary_is_masked] = -100
            
            # [Complementary Noisy, Clean]
            input_ids_complementary = torch.cat([complementary_noisy_input_ids, original_input_ids], dim=1)
            
            # Combine batches: [Batch_Primary; Batch_Complementary]
            input_ids = torch.cat([input_ids_primary, input_ids_complementary], dim=0)
            
            # Combine labels
            # Context part (Clean) has labels -100
            labels_context = torch.full_like(original_labels, -100)
            labels_primary = torch.cat([labels, labels_context], dim=1)
            labels_complementary = torch.cat([complementary_labels, labels_context], dim=1)
            
            labels = torch.cat([labels_primary, labels_complementary], dim=0)
            
        outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels if labels is not None else None, 
        )

        hidden_states = outputs[0]
        
        if labels is not None:
             # We only care about the first half (Noisy part) for prediction
             hidden_states = hidden_states[:, :hidden_states.shape[1]//2, :]
             labels = labels[:, :labels.shape[1]//2]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(lm_logits.reshape(-1, lm_logits.size(-1)), labels.reshape(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens=20, 
        mask_id=50256,
        block_size=32,
        top_p=0.95,
        temperature=1.0,
        threshold=0.9, # Confidence threshold
        verbose=False,
        tokenizer=None,
        **kwargs
    ):
        
        original_input_length = input_ids.shape[1]
        
        num_blocks = (max_new_tokens + block_size - 1) // block_size
        
        
        x_init = torch.full((input_ids.shape[0], max_new_tokens), mask_id, device=input_ids.device, dtype=torch.long)
        input_ids = torch.cat([input_ids, x_init], dim=1)
        
        x_t = input_ids.clone()
        
        current_len = original_input_length
        
        for i in range(num_blocks):
            start_idx = current_len
            end_idx = min(current_len + block_size, input_ids.shape[1])
            block_len = end_idx - start_idx
            
            step_count = 0
            while True:
                step_count += 1
                mask_idx = (x_t[:, start_idx:end_idx] == mask_id)
                if mask_idx.sum() == 0:
                    break
                
                
                outputs = self(x_t, return_dict=True)
                logits = outputs.logits
                
                block_logits = logits[:, start_idx:end_idx, :]
                
                x_1, p_1t = self.sample_with_top_p(block_logits, top_p=top_p, temperature=temperature)
                
                x1_p = torch.gather(p_1t, dim=-1, index=x_1.unsqueeze(-1)).squeeze(-1)
                
                x1_p = torch.where(mask_idx, x1_p, torch.tensor(-float('inf'), device=x1_p.device))
                
                unmask_idx = (x1_p > threshold)
                
                max_prob_idx = x1_p.argmax(dim=-1)
                for b_idx in range(unmask_idx.shape[0]):
                    unmask_idx[b_idx, max_prob_idx[b_idx]] = True
                
                current_block_tokens = x_t[:, start_idx:end_idx]
                current_block_tokens = torch.where(unmask_idx, x_1, current_block_tokens)
                x_t[:, start_idx:end_idx] = current_block_tokens
                
                if verbose and tokenizer:
                    generated_part = tokenizer.decode(x_t[0, original_input_length:])
                    clean_part = generated_part.replace(tokenizer.eos_token, '[EOS]').replace('\n', ' ')
                    print(f"Block {i+1}, Step {step_count}: {clean_part}")
            
            current_len = end_idx
            
        return x_t

    def sample_with_top_p(self, logits, top_p=0.95, temperature=1.0):
        if temperature > 0:
            logits = logits / temperature
            
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        probs[indices_to_remove] = 0
        probs = probs / probs.sum(dim=-1, keepdim=True)
        
        x_1 = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(probs.size(0), probs.size(1))
        
        return x_1, probs
