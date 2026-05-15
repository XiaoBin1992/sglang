# SPDX-License-Identifier: Apache-2.0
"""Diffusion paged KV cache pool initialization.

Mirrors sglang LLM startup flow (sglang/srt/model_executor/model_runner_kv_cache_mixin.py
+ sglang/srt/model_executor/pool_configurator.py):

  1. profile available GPU memory after model load
  2. compute per-token cell size from model shape
  3. derive max_total_num_tokens
  4. instantiate the LLM-side ``MHATokenToKVPool`` directly — its ``_create_buffers``
     already routes through ``_alloc_kv_cache`` which talks to the OmniFlow
     ``RequestCache`` when enabled, so the returned k/v buffers are
     IPC-shareable with the LLM side without any extra plumbing here.

Only the model-shape extraction (num_layers / num_heads / head_dim) and the
profile-budget formula are diffusion-specific; everything below the pool
construction line is borrowed from the LLM runtime.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch

from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.utils.common import get_available_gpu_memory

logger = logging.getLogger(__name__)


_DTYPE_STR_MAP = {
    "auto": None,  # follow model dtype
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp8": torch.float8_e4m3fn,
    "fp8_e4m3": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
}


@dataclass
class _ModelKVShape:
    num_layers: int
    num_heads: int
    head_dim: int
    v_head_dim: int


def _resolve_transformer(pipeline) -> torch.nn.Module:
    """Find the main DiT-style transformer on the pipeline.

    Diffusion pipelines expose the transformer under different attribute names
    depending on the model family.
    """
    for name in ("transformer", "video_dit", "audio_dit", "transformer_2"):
        getter = getattr(pipeline, "get_module", None)
        m = getter(name) if callable(getter) else getattr(pipeline, name, None)
        if m is not None:
            return m
    raise RuntimeError(
        "init_diffusion_kv_pool: could not locate a transformer on pipeline "
        f"(tried transformer / video_dit / audio_dit / transformer_2): {pipeline}"
    )


def _resolve_model_kv_shape(transformer) -> _ModelKVShape:
    """Pull (num_layers, num_heads, head_dim) from arch_config.

    The fields used here (``num_layers``, ``num_attention_heads``,
    ``attention_head_dim``) match what
    ``sglang/multimodal_gen/runtime/pipelines_core/stages/causal_denoising.py``
    already reads from ``transformer.config.arch_config`` for its own dense
    KV cache, so any model that runs causal denoising will work without
    further adapter code.
    """
    arch = transformer.config.arch_config
    if hasattr(arch, "num_layers") and hasattr(arch, "num_attention_heads"):
        num_layers = int(arch.num_layers)
        num_heads = int(arch.num_attention_heads)
        head_dim = int(arch.attention_head_dim)
    elif hasattr(arch, "num_blocks") and hasattr(arch, "num_heads"):
        num_layers = int(arch.num_blocks)
        num_heads = int(arch.num_heads)
        head_dim = int(arch.head_dim)
    else:
        raise NotImplementedError(
            f"init_diffusion_kv_pool: unknown arch_config shape: {type(arch).__name__}"
        )
    v_head_dim = int(getattr(arch, "v_head_dim", head_dim))
    return _ModelKVShape(
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        v_head_dim=v_head_dim,
    )


def _resolve_kv_dtype(server_args, fallback: torch.dtype) -> torch.dtype:
    name = (getattr(server_args, "kv_cache_dtype", "auto") or "auto").lower()
    mapped = _DTYPE_STR_MAP.get(name, fallback)
    return mapped if mapped is not None else fallback


def _resolve_attention_tp(server_args) -> tuple[int, int]:
    """Return (tp_size, tp_rank) for KV head sharding.

    Diffusion does not use sglang's dp_attention split, so attention tp ==
    overall tp_size. Falls back to (1, 0) for single-GPU.
    """
    tp_size = int(getattr(server_args, "tp_size", None) or 1)
    try:
        # 0 when distributed not yet initialized
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    except Exception:
        rank = 0
    tp_rank = rank % tp_size if tp_size > 0 else 0
    return tp_size, tp_rank


def _compute_cell_size(
    shape: _ModelKVShape,
    num_kv_heads_per_rank: int,
    dtype: torch.dtype,
) -> int:
    """Per-token KV cache cost in bytes, summed over all layers.

    Formula matches sglang ``DefaultPoolConfigurator._compute_cell_size`` for
    the non-MLA branch:
        cell_size = num_kv_heads * (head_dim + v_head_dim) * num_layers * dtype_size
    """
    elt = torch._utils._element_size(dtype)
    return (
        num_kv_heads_per_rank
        * (shape.head_dim + shape.v_head_dim)
        * shape.num_layers
        * elt
    )


def _profile_available_bytes(
    device: str,
    gpu_id: int,
    pre_pipeline_load_memory_gb: float,
    mem_fraction_static: float,
    distributed: bool,
) -> int:
    """Mirror ``ModelRunnerKVCacheMixin._profile_available_bytes``.

    rest_gb = post - pre * (1 - mem_fraction_static)
    """
    post = get_available_gpu_memory(
        device, gpu_id, distributed=distributed, empty_cache=True
    )
    rest_gb = post - pre_pipeline_load_memory_gb * (1 - mem_fraction_static)
    return int(rest_gb * (1 << 30))


def init_diffusion_kv_pool(
    *,
    server_args,
    pipeline,
    device: str,
    gpu_id: int,
    pre_pipeline_load_memory_gb: float,
    distributed: bool = False,
) -> Optional[MHATokenToKVPool]:
    """Profile leftover GPU memory and build a paged KV pool for diffusion.

    Returns the LLM-side ``MHATokenToKVPool`` instance directly. The caller
    typically attaches it to the pipeline (e.g. ``pipeline.kv_pool = pool``)
    so the paged attention impl can pull layer buffers via
    ``pool.k_buffer[layer_idx]`` / ``pool.v_buffer[layer_idx]``.

    Returns ``None`` when the feature is disabled, so callers can use the
    result as a soft switch.
    """
    if not getattr(server_args, "enable_paged_kv_cache", False):
        return None

    transformer = _resolve_transformer(pipeline)
    shape = _resolve_model_kv_shape(transformer)

    # dtype: fall back to the transformer's parameter dtype when "auto".
    try:
        param_dtype = next(transformer.parameters()).dtype
    except StopIteration:
        param_dtype = torch.bfloat16
    dtype = _resolve_kv_dtype(server_args, fallback=param_dtype)

    tp_size, tp_rank = _resolve_attention_tp(server_args)
    num_kv_heads_per_rank = max(1, shape.num_heads // tp_size)
    head_ids = list(
        range(
            tp_rank * num_kv_heads_per_rank,
            (tp_rank + 1) * num_kv_heads_per_rank,
        )
    )

    # ---- profile + budget ----
    cell_size = _compute_cell_size(shape, num_kv_heads_per_rank, dtype)
    available_bytes = _profile_available_bytes(
        device=device,
        gpu_id=gpu_id,
        pre_pipeline_load_memory_gb=pre_pipeline_load_memory_gb,
        mem_fraction_static=server_args.mem_fraction_static,
        distributed=distributed,
    )
    page_size = int(server_args.page_size)
    max_total_num_tokens = available_bytes // cell_size
    max_total_num_tokens = (max_total_num_tokens // page_size) * page_size

    # honor explicit user cap when provided
    cap = getattr(server_args, "max_total_num_tokens", None)
    if cap is not None and cap > 0:
        max_total_num_tokens = min(
            max_total_num_tokens, (cap // page_size) * page_size
        )

    if max_total_num_tokens <= 0:
        raise RuntimeError(
            "Not enough GPU memory for diffusion KV cache. "
            f"available={available_bytes / (1 << 30):.2f} GB, "
            f"cell_size={cell_size} B/token. "
            "Try lowering --mem-fraction-static or freeing more memory."
        )

    logger.info(
        "Diffusion KV pool: layers=%d, heads(tp)=%d, head_dim=%d, dtype=%s, "
        "page_size=%d, cell_size=%d B/token, available=%.2f GB → "
        "max_total_num_tokens=%d (%.2f GB)",
        shape.num_layers,
        num_kv_heads_per_rank,
        shape.head_dim,
        dtype,
        page_size,
        cell_size,
        available_bytes / (1 << 30),
        max_total_num_tokens,
        max_total_num_tokens * cell_size / (1 << 30),
    )

    # ---- build the pool. Reuses LLM-side MHATokenToKVPool verbatim:
    #      _create_buffers → _alloc_kv_cache → RequestCache.alloc_global_params
    #      when enable_request_cache is on, otherwise plain torch.zeros.
    pool = MHATokenToKVPool(
        size=max_total_num_tokens,
        page_size=page_size,
        dtype=dtype,
        head_num=num_kv_heads_per_rank,
        head_dim=shape.head_dim,
        v_head_dim=shape.v_head_dim,
        layer_num=shape.num_layers,
        device=device,
        enable_memory_saver=False,
        start_layer=0,
        end_layer=shape.num_layers - 1,
        enable_alt_stream=False,
        enable_kv_cache_copy=False,
        layer_ids=list(range(shape.num_layers)),
        head_ids=head_ids,
        sliding_window_size=None,
    )

    # Surface the budget so callers / metrics can read it back.
    pool.max_total_num_tokens = max_total_num_tokens
    return pool
