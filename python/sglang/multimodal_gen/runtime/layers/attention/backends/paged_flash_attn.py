# SPDX-License-Identifier: Apache-2.0
"""Paged-KV variant of FlashAttention for diffusion (OmniFlow).

Reuses the LLM-side primitives:

  - ``sglang.jit_kernel.flash_attention.flash_attn_with_kvcache`` — same kernel
    SGLang LLM uses for paged prefill/decode.
  - ``sglang.srt.mem_cache.memory_pool.MHATokenToKVPool`` — KV pool layout
    (one ``(size + page_size, head_num, head_dim)`` tensor per layer); we read
    via ``get_kv_buffer(layer_id)`` and write via ``set_kv_buffer(...,
    layer_id_override=...)``.
  - ``sglang.srt.mem_cache.allocator.OmniPagedTokenToKVPoolAllocator`` — the
    place that turns ``omni_flow.slots`` into token-level kv indices, so the
    builder below can reuse the same translation rules.

Diffusion-specific bits live only in the metadata builder
(``build_paged_metadata``) and the layer-id resolution from ``prefix``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, List, Optional

import torch

from sglang.jit_kernel.flash_attention import flash_attn_with_kvcache
from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

logger = logging.getLogger(__name__)


# ----------------------------- metadata -----------------------------


@dataclass
class PagedFlashAttentionMetadata(AttentionMetadata):
    """Carries everything ``flash_attn_with_kvcache`` needs.

    Lives on ``ForwardContext.attn_metadata`` for the duration of a forward
    pass; constructed once by the diffusion scheduler / sglang_backend before
    each step.
    """

    current_timestep: int = 0

    # KV pool reference (LLM-side MHATokenToKVPool instance).
    kv_pool: Any = None

    # Page table: (B, max_num_pages_per_seq) int32.
    page_table: Optional[torch.Tensor] = None
    # Per-batch total tokens already written in KV (= prefix_len + extend_len).
    cache_seqlens: Optional[torch.Tensor] = None  # int32 (B,)
    # Cumulative new-query lengths (B + 1,) int32.
    cu_seqlens_q: Optional[torch.Tensor] = None
    # Cumulative new-key lengths (B + 1,) int32.
    cu_seqlens_k: Optional[torch.Tensor] = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0

    # Token-level absolute KV indices for new tokens being written this step.
    # Same semantics as sglang ForwardBatch.out_cache_loc.
    out_cache_loc: Optional[torch.Tensor] = None  # int64

    # Per-layer flag — write once per (timestep, layer) so multiple attention
    # blocks don't double-write. Consumed by ``PagedFlashAttentionImpl.forward``.
    write_kv: bool = True

    # Optional sliding window (kept for parity with LLM forward_extend; None =
    # full attention).
    sliding_window_size: Optional[int] = None

    # Optional per-batch dummy_page_offset used by OmniPagedTokenToKVPoolAllocator
    # when sglang reserves padded slot-0 pages.
    dummy_page_offset: int = 0

    # Convenience handles.
    extra: dict = field(default_factory=dict)


# ----------------------------- builder -----------------------------


def _build_page_table_from_slots(
    slots_per_req: List[List[int]],
    dummy_page_offset: int,
    device: torch.device,
) -> torch.Tensor:
    bs = len(slots_per_req)
    if bs == 0:
        return torch.empty((0, 0), dtype=torch.int32, device=device)
    max_pages = max(len(s) for s in slots_per_req)
    out = torch.zeros((bs, max_pages), dtype=torch.int32, device=device)
    for i, s in enumerate(slots_per_req):
        if not s:
            continue
        ids = torch.tensor(s, dtype=torch.int32, device=device)
        if dummy_page_offset:
            ids = ids + dummy_page_offset
        out[i, : len(s)] = ids
    return out


def _build_out_cache_loc(
    slots_per_req: List[List[int]],
    prefix_lens: List[int],
    extend_lens: List[int],
    page_size: int,
    dummy_page_offset: int,
    device: torch.device,
) -> torch.Tensor:
    """Translate (slots, prefix_len, extend_len) into token-level KV indices
    for the new tokens being written this step. Same convention as
    ``OmniPagedTokenToKVPoolAllocator.alloc_extend``.
    """
    total = sum(extend_lens)
    out = torch.empty((total,), dtype=torch.int64)
    off = 0
    for slots, p, e in zip(slots_per_req, prefix_lens, extend_lens):
        for i in range(e):
            pos = p + i
            page_id = slots[pos // page_size] + dummy_page_offset
            out[off + i] = page_id * page_size + (pos % page_size)
        off += e
    return out.to(device, non_blocking=True)


def build_paged_metadata(
    *,
    kv_pool,
    slots_per_req: List[List[int]],
    prefix_lens: List[int],
    extend_lens: List[int],
    current_timestep: int = 0,
    sliding_window_size: Optional[int] = None,
    dummy_page_offset: int = 0,
    device: Optional[torch.device] = None,
) -> PagedFlashAttentionMetadata:
    """Construct a PagedFlashAttentionMetadata from external slot info.

    Inputs come from the OmniFlow scheduler / sglang_backend after it has
    called ``MemoryManagerClient.alloc_global_params_by_page_ids``.

    All tensors land on the KV pool's device by default.
    """
    if device is None:
        device = torch.device(kv_pool.device)
    page_size = kv_pool.page_size

    page_table = _build_page_table_from_slots(slots_per_req, dummy_page_offset, device)

    cache_seqlens_list = [p + e for p, e in zip(prefix_lens, extend_lens)]
    cache_seqlens = torch.tensor(cache_seqlens_list, dtype=torch.int32, device=device)

    cu_q = [0]
    cu_k = [0]
    for p, e in zip(prefix_lens, extend_lens):
        cu_q.append(cu_q[-1] + e)
        cu_k.append(cu_k[-1] + p + e)
    cu_seqlens_q = torch.tensor(cu_q, dtype=torch.int32, device=device)
    cu_seqlens_k = torch.tensor(cu_k, dtype=torch.int32, device=device)

    out_cache_loc = _build_out_cache_loc(
        slots_per_req, prefix_lens, extend_lens, page_size, dummy_page_offset, device
    )

    return PagedFlashAttentionMetadata(
        current_timestep=current_timestep,
        kv_pool=kv_pool,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max(extend_lens) if extend_lens else 0,
        max_seqlen_k=max(cache_seqlens_list) if cache_seqlens_list else 0,
        out_cache_loc=out_cache_loc,
        write_kv=True,
        sliding_window_size=sliding_window_size,
        dummy_page_offset=dummy_page_offset,
    )


# ----------------------------- impl -----------------------------


_BLOCK_PREFIX_RE = re.compile(r"\.blocks\.(\d+)\.")


def _layer_id_from_prefix(prefix: str) -> Optional[int]:
    """Extract layer index from a module prefix like
    ``transformer.blocks.5.attn1.attn``.
    """
    if not prefix:
        return None
    m = _BLOCK_PREFIX_RE.search(prefix)
    if m:
        return int(m.group(1))
    return None


class PagedFlashAttentionImpl(AttentionImpl):
    """Paged FlashAttention impl. KV is stored in an external MHATokenToKVPool;
    this impl writes the new K/V via ``out_cache_loc`` and reads through the
    page table — exactly mirroring sglang LLM ``forward_extend``.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: Optional[int] = None,
        prefix: str = "",
        layer_id: Optional[int] = None,
        **_,
    ) -> None:
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_size = head_size
        self.softmax_scale = softmax_scale
        self.causal = causal
        self.prefix = prefix
        # Resolve layer id once. Caller may also pass it explicitly.
        if layer_id is None:
            layer_id = _layer_id_from_prefix(prefix)
        if layer_id is None:
            raise ValueError(
                f"PagedFlashAttentionImpl: cannot infer layer_id from prefix={prefix!r}; "
                "pass layer_id=<int> via extra_impl_args."
            )
        self.layer_id = int(layer_id)

    # ------- write -------

    def _maybe_write_kv(
        self,
        meta: PagedFlashAttentionMetadata,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        if not meta.write_kv or meta.out_cache_loc is None:
            return
        # Reshape new K/V to (total_new_tokens, head, dim).
        k = key.reshape(-1, self.num_kv_heads, self.head_size)
        v = value.reshape(-1, self.num_kv_heads, self.head_size)
        # Reuse MHATokenToKVPool.set_kv_buffer — handles fp8 path, alt_stream, etc.
        meta.kv_pool.set_kv_buffer(
            layer=None,
            loc=meta.out_cache_loc,
            cache_k=k,
            cache_v=v,
            layer_id_override=self.layer_id,
        )

    # ------- forward -------

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: Optional[AttentionMetadata] = None,
        **_,
    ) -> torch.Tensor:
        if not isinstance(attn_metadata, PagedFlashAttentionMetadata):
            raise RuntimeError(
                "PagedFlashAttentionImpl requires PagedFlashAttentionMetadata "
                f"on the forward context, got {type(attn_metadata).__name__}."
            )
        meta = attn_metadata
        page_size = meta.kv_pool.page_size

        # 1. Write the new K/V into the paged pool first, so the kernel can
        #    read them via cache_seqlens that already include the extend.
        self._maybe_write_kv(meta, key, value)

        # 2. View per-layer pool tensor as (num_pages, page_size, H, D).
        k_buffer, v_buffer = meta.kv_pool.get_kv_buffer(self.layer_id)
        k_cache = k_buffer.view(-1, page_size, self.num_kv_heads, self.head_size)
        v_cache = v_buffer.view(-1, page_size, self.num_kv_heads, self.head_size)

        q = query.contiguous().reshape(-1, self.num_heads, self.head_size)

        window_size = (
            (meta.sliding_window_size, 0)
            if meta.sliding_window_size is not None
            else (-1, -1)
        )

        out = flash_attn_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=meta.page_table,
            cache_seqlens=meta.cache_seqlens,
            cu_seqlens_q=meta.cu_seqlens_q,
            cu_seqlens_k_new=meta.cu_seqlens_k,
            max_seqlen_q=meta.max_seqlen_q,
            softmax_scale=self.softmax_scale,
            causal=self.causal,
            window_size=window_size,
        )
        return out


# ----------------------------- backend -----------------------------


class PagedFlashAttentionMetadataBuilder(AttentionMetadataBuilder):
    def __init__(self) -> None:
        pass

    def prepare(self) -> None:
        pass

    def build(self, **kwargs: dict) -> PagedFlashAttentionMetadata:
        # Diffusion path constructs metadata externally via build_paged_metadata,
        # so the builder just returns an empty placeholder.
        return PagedFlashAttentionMetadata()


class PagedFlashAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = False

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.FA_PAGED

    @staticmethod
    def get_impl_cls() -> type:
        return PagedFlashAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type:
        return PagedFlashAttentionMetadata

    @staticmethod
    def get_builder_cls() -> type:
        return PagedFlashAttentionMetadataBuilder
