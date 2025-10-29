# coding=utf-8
# Copyright 2024 NUS Show Lab, HuggingFace.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import Tensor
import torch.nn.functional as F
from transformers import AutoConfig
from .modeling_utils import ConfigMixin, ModelMixin, register_to_config
from .sampling import cosine_schedule, mask_by_random_topk
from .phi import PhiForCausalLM
from .base import Transformer
from pathlib import Path
from training.utils import mask_or_random_replace_tokens
from training.prompting_utils import create_attention_mask_predict_next, create_attention_mask_for_mmu
from jaxtyping import Int
from einops import rearrange



class Showo(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
            self,
            vocab_size: int,
            max_seq_len: int,
            num_layers_input: int,
            num_layers_output: int,
            d_model: int,
            d_mlp: int,
            num_heads: int,
            image_len: int,
            dtype: str,
            accelerator,
    ):
        super().__init__()
        self.register_to_config(mask_token_id=vocab_size - 1)
        self.model = Transformer(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            num_layers=num_layers_input + num_layers_output,
            d_model=d_model,
            d_mlp=d_mlp,
            num_heads=num_heads,
            dtype=dtype,
        )
        self.accelerator = accelerator

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = True

    def assemble_attention_mask(
            self,
            input_ids,
            batch_size_t2i,
            batch_size_lm,
            batch_size_mmu,
            pad_id,
            soi_id,
            eoi_id,
    ):
        """
        Assemble attention masks for all three flows: t2i, lm, and mmu.

        Args:
            input_ids: Concatenated input_ids tensor [batch_size_t2i + batch_size_lm + batch_size_mmu, seq_length]
            batch_size_t2i: Batch size for text-to-image flow
            batch_size_lm: Batch size for language modeling flow
            batch_size_mmu: Batch size for multimodal understanding flow
            pad_id: Padding token ID
            soi_id: Start of image token ID
            eoi_id: End of image token ID

        Returns:
            attention_mask: Concatenated attention mask tensor for all flows
        """
        attention_masks = []

        # T2I attention mask
        if batch_size_t2i > 0:
            input_ids_t2i = input_ids[:batch_size_t2i]
            attention_mask_t2i = create_attention_mask_predict_next(
                input_ids_t2i,
                pad_id=pad_id,
                soi_id=soi_id,
                eoi_id=eoi_id,
                rm_pad_in_image=True,
                return_inverse_mask=True
            ).to(torch.bfloat16)
            attention_masks.append(attention_mask_t2i)

        # LM attention mask
        if batch_size_lm > 0:
            input_ids_lm = input_ids[batch_size_t2i:batch_size_t2i + batch_size_lm]
            attention_mask_lm = create_attention_mask_predict_next(
                input_ids_lm,
                pad_id=pad_id,
                soi_id=soi_id,
                eoi_id=eoi_id,
                return_inverse_mask=True
            ).to(torch.bfloat16)
            attention_masks.append(attention_mask_lm)

        # MMU attention mask
        if batch_size_mmu > 0:
            input_ids_mmu = input_ids[batch_size_t2i + batch_size_lm:]
            attention_mask_mmu = create_attention_mask_for_mmu(
                input_ids_mmu,
                eoi_id=eoi_id,
                return_inverse_mask=True
            ).to(torch.bfloat16)
            attention_masks.append(attention_mask_mmu)

        # Concatenate all attention masks
        attention_mask = torch.cat(attention_masks, dim=0)

        return attention_mask

    def forward(
            self,
            input_ids,
            config,
            input_embeddings=None,
            labels=None,
            label_smoothing=0.0,
            batch_size_t2i=0,
            batch_size_lm=0,
            batch_size_mmu=0,
            max_seq_length=128,
            labels_mask_text=None,
            labels_mask_image=None,
            global_step=None,
            pad_id=None,
            soi_id=None,
            eoi_id=None,
            mask_schedule=None,
            ignore_id=-100,
            **kwargs,
    ):

        assert input_embeddings is None

        # Apply masking to t2i tokens if labels is provided and we have batch_size_t2i > 0
        mask_prob = None
        if labels is not None and batch_size_t2i > 0:
            with torch.no_grad():
                # Extract t2i portion
                input_ids_t2i = input_ids[:batch_size_t2i]
                labels_t2i = labels[:batch_size_t2i]

                # Apply masking
                input_ids_t2i, labels_t2i, loss_weight, mask_prob = mask_or_random_replace_tokens(
                    input_ids_t2i,
                    labels_t2i,
                    self.config.mask_token_id,
                    soi_id,
                    eoi_id,
                    config,
                    mask_schedule,
                    ignore_id,
                )

                # Replace t2i portion in input_ids and labels
                input_ids = torch.cat([input_ids_t2i, input_ids[batch_size_t2i:]], dim=0)
                labels = torch.cat([labels_t2i, labels[batch_size_t2i:]], dim=0)

        # Assemble attention mask for all flows
        attn_mask = self.assemble_attention_mask(
            input_ids=input_ids,
            batch_size_t2i=batch_size_t2i,
            batch_size_lm=batch_size_lm,
            batch_size_mmu=batch_size_mmu,
            pad_id=pad_id,
            soi_id=soi_id,
            eoi_id=eoi_id,
        )

        logits = self.model(input_ids=input_ids, attn_mask=attn_mask)

        torch.save(input_ids, "input_ids.pt")
        torch.save(labels, "labels.pt")
        torch.save(attention_mask, "attention_mask.pt")

        if labels is not None:
            # 1. Mask token prediction (discrete diffusion) for image generation
            # Note that, max_seq_length indicates the maximum number of text tokens, maybe a bit confused.

            # Note that for masked image prediction, we assume that each
            # position generates a prediction at the same position, not the
            # next. There are text tokens in these sequences, but these are set
            # to ignore in the labels.
            assert torch.all(labels[:batch_size_t2i][:, torch.argmax(labels[:batch_size_t2i] == -100) + 1] == soi_id)
            loss_t2i = F.cross_entropy(
                logits[:batch_size_t2i].contiguous().view(-1, self.config.vocab_size),
                labels[:batch_size_t2i].contiguous().view(-1), ignore_index=-100,
            )

            # 2. Next token prediction for language modeling
            loss_lm = F.cross_entropy(
                logits[batch_size_t2i:batch_size_t2i + batch_size_lm, :-1].contiguous().view(-1, self.config.vocab_size),
                labels[batch_size_t2i:batch_size_t2i + batch_size_lm, 1:].contiguous().view(-1), ignore_index=-100,
            )

            # 3. Next token prediction for captioning/multimodal understanding
            assert torch.all(labels[-batch_size_mmu:, : 3 + self.config.image_len] == -100)
            loss_mmu = F.cross_entropy(
                logits[-batch_size_mmu:, :-1].contiguous().view(-1, self.config.vocab_size),
                labels[-batch_size_mmu:, 1:].contiguous().view(-1), ignore_index=-100,
            )

            return logits, loss_t2i, loss_lm, loss_mmu

        return logits
    
    def predict_t2i_with_remask_and_labels(
        self,
        input_ids: Int[Tensor, "batch seq"],
        soi_id: int,
        pad_id: int,
        eoi_id: int,
        temperature=1.0,
        timesteps=5,
        noise_schedule=cosine_schedule,
        generator: torch.Generator = None,
        config=None,
        **kwargs,
    ) -> Float[Tensor, "batch seq_image d_vocab"]:
        """
        Input ids includes both a prompt (text) and image

        Only returns predictions for the image tokens.
        """
        batch_size = input_ids.shape[0]

        mask_token_id = self.config.mask_token_id
        image_len_tokens = self.config.image_len
        assert input_ids.shape[1] > image_len_tokens
        num_new_special_tokens = config.model.tokenize.num_new_special_tokens

        # Extract image tokens
        image_start_idx = torch.argmax(input_ids == soi_id, dim=1) + 1

        # Mask image tokens in full inputs tensor
        input_ids_masked = input_ids.clone()
        input_ids_masked[:, image_start_idx : image_start_idx + image_len_tokens] = mask_token_id

        # Create image logits tensor that we will iteratively fill in and return
        logits_image_all_iters = torch.empty(
            (batch_size, image_len_tokens, self.config.vocab_size),
            dtype=float, 
            device=input_ids.device
        )

        attention_mask = self.assemble_attention_mask(
            input_ids=input_ids,
            batch_size_t2i=batch_size,
            batch_size_lm=0,
            batch_size_mmu=0,
            pad_id=pad_id,
            soi_id=soi_id,
            eoi_id=eoi_id,
        )

        for step in range(timesteps):
            # Run forward pass on entire sequence
            logits = self(input_ids_masked, attention_mask=attention_mask)
            logits_image = logits[:, image_start_idx : image_start_idx + image_len_tokens]

            # Sample from image token distributions, just so that we can compute
            # the remasking tendency as the probability of the sampled token
            probs = logits_image.softmax(dim=-1)
            sampled_ids = torch.multinomial(
                rearrange(probs, "batch seq d_vocab -> (batch seq) d_vocab"), 
                1, 
                generator=generator,
            )[:, 0]
            sampled_ids = rearrange(sampled_ids, "(batch seq) -> batch seq", batch=batch_size)

            # Set tokens that have already been unmasked to previously-sampled
            # value
            mask_curr = input_ids_masked[:, image_start_idx : image_start_idx + image_len_tokens] == mask_token_id
            sampled_ids = torch.where(
                mask_curr,
                sampled_ids,
                input_ids_masked[:, image_start_idx : image_start_idx + image_len_tokens]
            )

            # Get the mask ratio for the next round. This is the ratio of masked
            # tokens over the whole image after we have remasked our predictions
            # for this round.
            ratio = 1.0 * (step + 1) / timesteps
            mask_ratio = noise_schedule(torch.tensor(ratio))

            # Computes the probabilities of each selected tokens.
            selected_probs = torch.gather(probs, -1, sampled_ids.long().unsqueeze(-1))
            selected_probs = selected_probs.squeeze(-1)

            # Sets probs for tokens predicted in previous rounds to inf so that
            # they don't get remasked when we remask by certainty
            selected_probs = torch.where(mask_curr, selected_probs, torch.finfo(selected_probs.dtype).max)

            # Get masking length for next round for each sample
            mask_len = (image_len_tokens * mask_ratio).floor().unsqueeze(0).to(logits.device)
            # Keeps at least one of prediction in this round and also masks out
            # at least one token for next round
            mask_len = torch.max(
                # Max: One token next round
                torch.tensor([1], device=logits.device),
                torch.min(
                    # Min: Keep at least one prediction from this round
                    mask_curr.sum(dim=-1, keepdim=True) - 1, 
                    mask_len,
                )
            )

            temperature = temperature * (1.0 - ratio)
            mask_next = mask_by_random_topk(mask_len, selected_probs, temperature, generator=generator)

            this_round_predictions = mask_curr & ~mask_next

            # Unmask tokens that were predicted this round using ground truth value
            input_ids_masked[:, image_start_idx : image_start_idx + image_len_tokens] = torch.where(
                this_round_predictions,
                input_ids[:, image_start_idx : image_start_idx + image_len_tokens],
                mask_token_id,
            )

            # Save logits for tokens that were predicted this round
            logits_image_all_iters = torch.where(
                this_round_predictions,
                logits_image,
                logits_image_all_iters,
            )


        return logits_image_all_iters