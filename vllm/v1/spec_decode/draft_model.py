# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.speculative import SpeculativeConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.spec_decode.eagle import EagleProposer

logger = init_logger(__name__)


class DraftModelProposer(EagleProposer):
    """Proposer that uses a standalone draft model instead of EAGLE heads."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(
            vllm_config=vllm_config,
            device=device,
            pass_hidden_states_to_model=False,
            runner=runner,
        )
        self._raise_if_multimodal()
        self._raise_if_mrope()
        self._raise_if_vocab_size_mismatch()
        self._raise_if_draft_tp_mismatch()

    def _raise_if_multimodal(self) -> None:
        if self.supports_mm_inputs:
            raise NotImplementedError(
                "Speculative decoding with draft models does not support "
                "multimodal models yet."
            )

    def _raise_if_mrope(self) -> None:
        if self.draft_model_config.uses_mrope:
            raise NotImplementedError(
                "Speculative decoding with draft models does not support "
                "M-RoPE yet."
            )

    def _raise_if_vocab_size_mismatch(self) -> None:
        self.vllm_config.speculative_config.verify_equal_vocab_size_if_draft_model()

    def _raise_if_draft_tp_mismatch(self) -> None:
        # For now we require draft TP to match target TP in this branch.
        spec_cfg: SpeculativeConfig = self.vllm_config.speculative_config
        tgt_tp = spec_cfg.target_parallel_config.tensor_parallel_size
        draft_tp = spec_cfg.draft_parallel_config.tensor_parallel_size
        if draft_tp != tgt_tp:
            raise ValueError(
                "Currently, 'draft_tensor_parallel_size' and target "
                "'tensor_parallel_size' must be the same for draft_model "
                f"spec decode. Got {draft_tp} and {tgt_tp}."
            )

    def load_model(self, target_model: Any) -> None:
        """Load draft model and register its attention layers as drafting layers."""
        del target_model

        target_attn_layer_names = set(
            get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase).keys()
        )

        from vllm.compilation.backends import set_model_tag

        with set_model_tag("draft_model"):
            model_config = self.vllm_config.speculative_config.draft_model_config
            assert model_config is not None
            loader = get_model_loader(self.vllm_config.load_config)
            load_config = self.vllm_config.load_config
            device_config = self.vllm_config.device_config
            load_device = (
                device_config.device
                if load_config.device is None
                else load_config.device
            )
            target_device = torch.device(load_device)
            # Initialize the draft model using draft model_config on the shared
            # runtime vllm_config so draft layers are registered into the same
            # forward context namespace, but with a unique prefix.
            original_model_config = self.vllm_config.model_config
            original_quant_config = self.vllm_config.quant_config
            # Use a dedicated model prefix to avoid collisions with target
            # layer names in compilation_config.static_forward_context.
            try:
                self.vllm_config.model_config = model_config
                self.vllm_config.quant_config = VllmConfig.get_quantization_config(
                    model_config, self.vllm_config.load_config
                )
                with set_default_torch_dtype(model_config.dtype):
                    with target_device:
                        self.model = initialize_model(
                            vllm_config=self.vllm_config,
                            model_config=model_config,
                            prefix="draft_model",
                        )
                    loader.load_weights(self.model, model_config)
                    process_weights_after_loading(
                        self.model, model_config, target_device
                    )
                    self.model = self.model.eval()
            finally:
                self.vllm_config.model_config = original_model_config
                self.vllm_config.quant_config = original_quant_config

        draft_attn_layer_names = (
            get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase).keys()
            - target_attn_layer_names
        )
        self.attn_layer_names = list(draft_attn_layer_names)
        if not self.attn_layer_names:
            raise RuntimeError(
                "Failed to find draft model attention layers in the runtime "
                "forward context."
            )
        self.indexer_layer_names = []
        self.draft_indexer_metadata_builder = None
