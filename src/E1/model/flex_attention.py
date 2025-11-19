from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypedDict

try:
    import torch
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

    if torch.cuda.is_available():
        flex_attention = torch.compile(flex_attention, dynamic=True)
except ImportError:
    # 在当前环境（如 torch 2.4.x）中可能不存在 torch.nn.attention.flex_attention。
    # 为了兼容，我们显式禁用 flex attention，并提供占位符类型 / 函数。
    flex_attention = None

    BlockMask = Any  # type: ignore[assignment]

    def create_block_mask(*args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        raise ImportError("Flex Attention is not available in this environment")


class FlexAttentionArgs(TypedDict, total=False):
    block_mask: BlockMask | None
    score_mod: Callable | None


def is_flex_attention_available() -> bool:
    """当前环境是否支持 flex attention。"""
    return flex_attention is not None


def create_block_causal_mask_optimized(sequence_ids: "torch.Tensor") -> BlockMask | None:
    """
    基于 sequence_ids 构建 block causal mask。

    当 flex attention 不可用时，返回 None，调用方需要避免走 flex 路径。
    """
    if not is_flex_attention_available():
        return None

    # Assumes sequence_ids is sorted in increasing order for each batch item, except for
    # the -1 values, which are used to indicate the padding tokens.
    def document_mask(b, h, q_idx, kv_idx):  # type: ignore[no-untyped-def]
        return (
            (sequence_ids[b, q_idx] >= sequence_ids[b, kv_idx])
            & (sequence_ids[b, q_idx] != -1)
            & (sequence_ids[b, kv_idx] != -1)
        )

    batch_size, seqlen = sequence_ids.shape
    return create_block_mask(document_mask, batch_size, 1, seqlen, seqlen, device=sequence_ids.device)


def flex_attention_func(
    query_states: "torch.Tensor",  # (bs, seqlen, nh, hs)
    key_states: "torch.Tensor",  # (bs, seqlen, nkv, hs)
    value_states: "torch.Tensor",  # (bs, seqlen, nkv, hs)
    score_mod: Callable | None = None,
    block_mask: BlockMask | None = None,
) -> "torch.Tensor":
    assert flex_attention is not None, "Flex Attention is not available in this environment"
    assert score_mod is None, "Score mod is not supported yet"
    query_states = query_states.transpose(1, 2).contiguous()  # (bs, nh, seqlen, hs)
    key_states = key_states.transpose(1, 2).contiguous()  # (bs, nkv, seqlen, hs)
    value_states = value_states.transpose(1, 2).contiguous()  # (bs, nkv, seqlen, hs)

    outputs = flex_attention(
        query_states,
        key_states,
        value_states,
        block_mask=block_mask,
        score_mod=score_mod,
        enable_gqa=query_states.shape[1] != key_states.shape[1],  # if nkv != nh
    )

    outputs = outputs.transpose(1, 2)  # (bs, seqlen, nh, hs)
    return outputs
