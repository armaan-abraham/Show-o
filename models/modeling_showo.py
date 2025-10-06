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
import torch.nn.functional as F
from transformers import AutoConfig
from .modeling_utils import ConfigMixin, ModelMixin, register_to_config
from .sampling import cosine_schedule, mask_by_random_topk
from .phi import PhiForCausalLM
from .base import Transformer
from pathlib import Path
from training.utils import mask_or_random_replace_tokens
from training.prompting_utils import create_attention_mask_predict_next, create_attention_mask_for_mmu



class Showo(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
            self,
            w_clip_vit,
            vocab_size,
            llm_vocab_size,
            accelerator,
            llm_model_path='',
            codebook_size=8192,
            num_vq_tokens=256,
            load_from_showo=True,
            **kwargs,
    ):
        super().__init__()
        assert not w_clip_vit
        self.vocab_size = vocab_size
        self.register_to_config(mask_token_id=vocab_size - 1)
        self.model = Transformer(vocab_size)
        self.output_size = self.vocab_size
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
            config=None,
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
        attention_mask = self.assemble_attention_mask(
            input_ids=input_ids,
            batch_size_t2i=batch_size_t2i,
            batch_size_lm=batch_size_lm,
            batch_size_mmu=batch_size_mmu,
            pad_id=pad_id,
            soi_id=soi_id,
            eoi_id=eoi_id,
        )

        # Save tensors on first step if main process
        if global_step == 0 and self.accelerator.is_main_process:
            print("saved input ids and attention mask")
            torch.save(input_ids, Path(__file__).parent / 'input_ids.pt')
            torch.save(attention_mask, Path(__file__).parent / 'attention_mask.pt')

        logits = self.model(input_ids=input_ids, attention_mask=attention_mask)['logits']

        if labels is not None:
            # 1. Mask token prediction (discrete diffusion) for image generation
            # Note that, max_seq_length indicates the maximum number of text tokens, maybe a bit confused.
            loss_t2i = F.cross_entropy(
                logits[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1, self.output_size),
                labels[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1), ignore_index=-100,
            )

            # 2. Next token prediction for language modeling
            loss_lm = F.cross_entropy(
                logits[batch_size_t2i:batch_size_t2i + batch_size_lm, :-1].contiguous().view(-1, self.output_size),
                labels[batch_size_t2i:batch_size_t2i + batch_size_lm, 1:].contiguous().view(-1), ignore_index=-100,
            )

            # 3. Next token prediction for captioning/multimodal understanding
            loss_mmu = F.cross_entropy(
                logits[-batch_size_mmu:, :-1].contiguous().view(-1, self.output_size),
                labels[-batch_size_mmu:, 1:].contiguous().view(-1), ignore_index=-100,
            )

            return logits, loss_t2i, loss_lm, loss_mmu, mask_prob

        return logits
