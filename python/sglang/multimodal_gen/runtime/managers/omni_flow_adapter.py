# SPDX-License-Identifier: Apache-2.0
"""Bridge OmniFlow → multimodal_gen paged attention.

Diffusion does not parse ``input_extra_infos["omni_flow"]`` natively; this
helper consumes the same dict the LLM side (``OmniFlowRadixCache`` /
``OmniPagedTokenToKVPoolAllocator``) reads from, and produces a
``PagedFlashAttentionMetadata`` ready to be dropped into ``Req.extra``.

Callers (OmniFlow scheduler / sglang_backend wrapper) typically do:

    from sglang.multimodal_gen.runtime.managers.omni_flow_adapter import (
        attach_paged_metadata_from_omni_flow,
    )

    attach_paged_metadata_from_omni_flow(
        req=multimodal_req,
        kv_pool=pipeline.kv_pool,
        omni_flow_info=input_extra_infos["omni_flow"],
        device=pipeline.kv_pool.device,
    )

This keeps the multimodal pipeline ignorant of OmniFlow's wire format —
``causal_denoising`` only sees a generic ``Req.extra["paged_attn_metadata"]``.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.paged_flash_attn import (
    PagedFlashAttentionMetadata,
    build_paged_metadata,
)

logger = logging.getLogger(__name__)


def build_metadata_from_omni_flow(
    *,
    kv_pool,
    omni_flow_info: dict,
    current_timestep: int = 0,
    sliding_window_size: Optional[int] = None,
    dummy_page_offset: int = 0,
    device: Optional[torch.device] = None,
) -> PagedFlashAttentionMetadata:
    """Translate one OmniFlow ``input_extra_infos["omni_flow"]`` payload into
    a paged attention metadata object.

    Expected fields on ``omni_flow_info`` (matches what
    ``omni_flow/compute_flow/llm/sglang_backend.py`` writes today):

        prefix_len:  int     # cached tokens (may not be page-aligned)
        extend_len:  int     # new tokens to write this step
        slots:       List[int]   # full slot list, length = ceil(seq_total / page_size)
    """
    prefix_len = int(omni_flow_info["prefix_len"])
    extend_len = int(omni_flow_info["extend_len"])
    slots = list(omni_flow_info["slots"])

    return build_paged_metadata(
        kv_pool=kv_pool,
        slots_per_req=[slots],
        prefix_lens=[prefix_len],
        extend_lens=[extend_len],
        current_timestep=current_timestep,
        sliding_window_size=sliding_window_size,
        dummy_page_offset=dummy_page_offset,
        device=device,
    )


def attach_paged_metadata_from_omni_flow(
    *,
    req,
    kv_pool,
    omni_flow_info: dict,
    current_timestep: int = 0,
    sliding_window_size: Optional[int] = None,
    dummy_page_offset: int = 0,
    device: Optional[torch.device] = None,
) -> PagedFlashAttentionMetadata:
    """Build paged metadata and stuff it into ``req.extra``.

    Returns the metadata so callers can also keep a reference if needed.
    """
    meta = build_metadata_from_omni_flow(
        kv_pool=kv_pool,
        omni_flow_info=omni_flow_info,
        current_timestep=current_timestep,
        sliding_window_size=sliding_window_size,
        dummy_page_offset=dummy_page_offset,
        device=device,
    )

    if not hasattr(req, "extra") or req.extra is None:
        # Req is a dataclass with default_factory=dict, so this branch only
        # fires for unusual callers; create the dict so we don't fail.
        try:
            req.extra = {}
        except Exception as e:
            raise TypeError(
                "attach_paged_metadata_from_omni_flow: req has no writable "
                f".extra attribute: {e}"
            )

    req.extra["paged_attn_metadata"] = meta
    return meta
