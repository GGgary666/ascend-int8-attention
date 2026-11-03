"""Real vLLM Ascend BF16 sharedkv operator, with metadata prepared separately."""

import torch
import torch_npu  # noqa: F401


def load_ops():
    from vllm_ascend.utils import bootstrap_custom_op_env

    bootstrap_custom_op_env(include_vendor_lib=True)
    import vllm_ascend.vllm_ascend_C  # noqa: F401


def prepare(case, *, max_query_length, max_seq_length):
    """Maxima are CPU scheduler values, never read back during graph capture."""
    load_ops()
    metadata = torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata(
        num_heads_q=case.q.shape[1], num_heads_kv=1, head_dim=512,
        cu_seqlens_q=case.cu_q, seqused_kv=case.seq_lens,
        batch_size=case.seq_lens.numel(), max_seqlen_q=max_query_length,
        max_seqlen_kv=max_seq_length,
        ori_topk=case.ori_indices.shape[-1] if case.ori_indices is not None else 0,
        cmp_topk=case.cmp_indices.shape[-1] if case.cmp_indices is not None else 0,
        cmp_ratio=case.ratio, ori_mask_mode=0 if case.ori_indices is not None else 4, cmp_mask_mode=3,
        ori_win_left=case.window - 1, ori_win_right=0,
        layout_q="TND", layout_kv="PA_ND", has_ori_kv=True,
        has_cmp_kv=case.cmp is not None, device=str(case.q.device),
    )
    return metadata


def run(case, metadata, *, return_lse=False):
    return torch.ops._C_ascend.npu_sparse_attn_sharedkv(
        case.q, ori_kv=case.ori, cmp_kv=case.cmp,
        ori_sparse_indices=case.ori_indices,
        cmp_sparse_indices=case.cmp_indices,
        ori_block_table=case.ori_block_table, cmp_block_table=case.cmp_block_table,
        cu_seqlens_q=case.cu_q, seqused_kv=case.seq_lens, sinks=case.sink,
        metadata=metadata, softmax_scale=case.softmax_scale, cmp_ratio=case.ratio,
        # The installed operator requires mode 4 even for explicit slots;
        # its metadata builder accepts mode 0, as used by vLLM DSpark.
        ori_mask_mode=4, cmp_mask_mode=3, ori_win_left=case.window - 1,
        ori_win_right=0, layout_q="TND", layout_kv="PA_ND",
        return_softmax_lse=return_lse,
    )
