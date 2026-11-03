"""Executable D=512 shared-KV contract, independent of Ascend kernels.

SWA and compressed keys participate in ONE softmax. Sink has a logit but no
value contribution. RoPE has already been applied to the last 64 dimensions.
"""

from dataclasses import dataclass
import math

import torch


DIM = 512


@dataclass(frozen=True)
class CacheQuantization:
    """Frozen symmetric per-channel scales shared by K and V.

    Scales are calibrated before cache population. They never change when a
    token is appended. Each cache kind may have its own scale vector.
    """

    scale: torch.Tensor

    def __post_init__(self):
        if self.scale.shape != (DIM,) or self.scale.dtype != torch.float32:
            raise ValueError("scale must be float32[512]")
        if not bool(torch.isfinite(self.scale).all() and (self.scale > 0).all()):
            raise ValueError("scale must be finite and strictly positive")

    @classmethod
    def calibrate(cls, samples):
        maximum = torch.zeros(DIM, dtype=torch.float32)
        for sample in samples:
            maximum = torch.maximum(maximum, sample.float().reshape(-1, DIM).abs().amax(0).cpu())
        return cls((maximum / 127).clamp_min(1e-8))

    @classmethod
    def from_msmodelslim(cls, tensors, prefix):
        scale = tensors[f"{prefix}.kv_cache_scale"].float().reshape(-1)
        offset = tensors[f"{prefix}.kv_cache_offset"].float().reshape(-1)
        if torch.count_nonzero(offset):
            raise ValueError("Protocol v1 requires symmetric C8; nonzero offsets are unsupported")
        return cls(scale)

    @property
    def inverse_scale(self):
        # Prepare once in FP64, rounded to FP32, before cache population.
        return self.scale.double().reciprocal().float()

    def quantize(self, x):
        return torch.round(x.float() * self.inverse_scale.to(x.device)).clamp(-127, 127).to(torch.int8)

    def dequantize(self, x):
        return x.float() * self.scale.to(x.device)


@dataclass
class SharedKVCase:
    q: torch.Tensor
    ori: torch.Tensor
    ori_block_table: torch.Tensor
    cu_q: torch.Tensor
    seq_lens: torch.Tensor
    sink: torch.Tensor
    cmp: torch.Tensor | None = None
    cmp_block_table: torch.Tensor | None = None
    cmp_indices: torch.Tensor | None = None
    ratio: int = 1
    window: int = 128
    softmax_scale: float = 1 / math.sqrt(DIM)
    ori_indices: torch.Tensor | None = None

    def to(self, device):
        return SharedKVCase(**{
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in vars(self).items()
        })


def gather(cache, table, batch, indices):
    block = cache.shape[1]
    pages = table[batch, indices // block].long()
    return cache[pages, indices % block, 0, :]


def gather_slots(cache, slots):
    """Original sparse indices use physical slots, with no block-table lookup."""
    return cache[slots // cache.shape[1], slots % cache.shape[1], 0, :]


def selected_rows(case):
    """Yield query row and independently paged original/compressed selections."""
    if case.ratio not in (1, 4, 128):
        raise ValueError("Supported compression ratios are 1, 4, 128")
    if case.q.shape[-1] != DIM or case.ori.shape[-2:] != (1, DIM):
        raise ValueError("Expected Q[T,H,512] and cache[P,B,1,512]")
    if case.ratio != 1 and (case.cmp is None or case.cmp_block_table is None):
        raise ValueError("Compressed attention requires a compressed cache and block table")
    for batch in range(case.seq_lens.numel()):
        begin, end = int(case.cu_q[batch]), int(case.cu_q[batch + 1])
        length = int(case.seq_lens[batch])
        if end - begin > length:
            raise ValueError("Query length cannot exceed sequence length")
        for t in range(begin, end):
            position = length - (end - begin) + (t - begin)
            ori_idx = torch.arange(max(0, position - case.window + 1), position + 1,
                                   device=case.ori.device)
            if case.ori_indices is None:
                ori = gather(case.ori, case.ori_block_table, batch, ori_idx)
            else:
                slots = case.ori_indices[t, 0].long()
                slots = slots[(slots >= 0) & (slots < case.ori.shape[0] * case.ori.shape[1])]
                ori = gather_slots(case.ori, slots)
            cmp = None
            if case.ratio != 1:
                # Production SCFA uses (position + 1) / cmpRatio as the bound.
                count = (position + 1) // case.ratio
                if case.ratio == 4 and case.cmp_indices is not None:
                    idx = case.cmp_indices[t, 0].long()
                    idx = idx[(idx >= 0) & (idx < count)]
                else:
                    idx = torch.arange(count, device=case.cmp.device)
                cmp = gather(case.cmp, case.cmp_block_table, batch, idx)
            yield t, ori, cmp


def sharedkv_reference(case):
    outputs, lses = [], []
    for t, ori, cmp in selected_rows(case):
        kv = ori if cmp is None else torch.cat((ori, cmp), 0)
        logits = case.q[t].double() @ kv.double().T * case.softmax_scale
        extended = torch.cat((logits, case.sink.double()[:, None]), dim=-1)
        lse = torch.logsumexp(extended, dim=-1)
        out = torch.exp(logits - lse[:, None]) @ kv.double()
        outputs.append(out.float())
        lses.append(lse.float())
    return torch.stack(outputs), torch.stack(lses)


def quantized_reference(case, ori_quant, cmp_quant=None):
    """INT8 QK and PV protocol; integer products are accumulated exactly on CPU.

    case caches are INT8, Q is floating point. Separate cache scales are
    supported. P is quantized against the shared maximum across both sets.
    """
    outputs = []
    for t, ori, cmp in selected_rows(case):
        groups = [(ori, ori_quant)]
        if cmp is not None:
            if cmp_quant is None:
                raise ValueError("Missing compressed-cache quantization parameters")
            groups.append((cmp, cmp_quant))
        logits = []
        for kv, quant in groups:
            scaled_q = case.q[t].float() * quant.scale.to(case.q.device)
            alpha = (scaled_q.abs().amax(-1, keepdim=True) / 127).clamp_min(1e-12)
            q8 = torch.round(scaled_q / alpha).clamp(-127, 127).to(torch.int32)
            dots = q8.cpu() @ kv.cpu().to(torch.int32).T
            logits.append(dots.to(case.q.device).float() * alpha * case.softmax_scale)
        scores = torch.cat(logits, dim=-1)
        maximum = scores.amax(-1, keepdim=True)
        lse = torch.logsumexp(torch.cat((scores, case.sink[:, None]), -1), -1)
        inverse_sum = torch.exp(maximum.squeeze(-1) - lse)
        out = torch.zeros_like(case.q[t], dtype=torch.float32)
        for score, (kv, quant) in zip(logits, groups):
            p8 = torch.round(torch.exp(score - maximum) * 127).clamp(0, 127).to(torch.int32)
            pv = p8.cpu() @ kv.cpu().to(torch.int32)
            out += pv.to(out.device).float() * quant.scale.to(out.device) / 127
        outputs.append(out * inverse_sum[:, None])
    return torch.stack(outputs)


def joint_quantized_row(q, caches, quantizers, sink, softmax_scale, logit_biases=None):
    """CPU oracle: one softmax over independently encoded INT8 KV kinds.

    Integer products and both residuals are explicit; no device tiling,
    padded matrices, scratch layout or synchronization is reproduced here.
    """
    logits = []
    for kind, (kv, quant) in enumerate(zip(caches, quantizers)):
        folded = q.float() * quant.scale
        alpha = (folded.abs().amax(-1, keepdim=True) * (1.0 / 127)).clamp_min(1e-12)
        normalized = (folded * alpha.reciprocal()).clamp(-127, 127)
        high = normalized.round().int()
        low = ((normalized - high.float()) * 127).round().int()
        dot = (high @ kv.int().T).float() + (low @ kv.int().T).float() * (1.0 / 127)
        logit = dot * (alpha * softmax_scale)
        if logit_biases is not None and logit_biases[kind] is not None:
            logit = logit + logit_biases[kind]
        logits.append(logit)
    scores = torch.cat(logits, dim=-1)
    maximum = torch.maximum(scores.amax(-1), sink)
    weights = torch.exp(scores - maximum[:, None])
    denominator = torch.exp(sink - maximum) + weights.sum(-1)
    high = (weights * 127).round().int()
    low = ((weights * 127 - high.float()) * 127).round().int()
    output = torch.zeros_like(q, dtype=torch.float32)
    offset = 0
    for kv, quant in zip(caches, quantizers):
        end = offset + len(kv)
        dot = (high[:, offset:end] @ kv.int()).float()
        dot += (low[:, offset:end] @ kv.int()).float() * (1.0 / 127)
        output += dot * (quant.scale * (1.0 / 127))
        offset = end
    return (output / denominator[:, None]).bfloat16(), maximum + denominator.log()


def tiled_quantized_reference(case, ori_quant, cmp_quant=None, tile_size=128, splits=1, query_group=1, return_lse=False,
                              residual_quantization=None, ori_row_factors=None, dense_c4=False, cmp_row_factors=None,
                              joint_dense=False, joint_sparse=False, joint_tile_limit=256):
    """Independent CPU model of online INT8 attention, retaining sparse holes.

    Uses exact integer QK/PV with FP32 online softmax and BF16 final rounding.
    This separates implementation errors from the intentional quantization
    error measured against sharedkv_reference().
    """
    if case.q.device.type != "cpu":
        raise ValueError("The integer oracle runs on CPU")
    if residual_quantization is None:
        residual_quantization = case.q.shape[1] <= 8
    result, lses = [], []
    for batch in range(case.seq_lens.numel()):
        first, last = int(case.cu_q[batch]), int(case.cu_q[batch + 1])
        length = int(case.seq_lens[batch])
        for t in range(first, last):
            position = length - last + t
            group_first = first + (t - first) // query_group * query_group
            group_last = min(last, group_first + query_group) - 1
            first_pos, last_pos = length - last + group_first, length - last + group_last
            if case.ori_indices is None:
                original = torch.arange(max(0, first_pos + 1 - case.window), last_pos + 1)
                original_valid = (original >= max(0, position + 1 - case.window)) & (original <= position)
            else:
                if query_group != 1:
                    raise ValueError("Explicit original slots require ungrouped queries")
                original = case.ori_indices[t, 0].long()
                original_valid = (original >= 0) & (original < case.ori.shape[0] * case.ori.shape[1])
            groups = [(case.ori, case.ori_block_table, original, original_valid, ori_quant)]
            cmp_multiplicity = None
            if case.ratio != 1:
                count = (position + 1) // case.ratio
                indices = case.cmp_indices[t, 0].long() if case.ratio == 4 else torch.arange((last_pos + 1) // case.ratio)
                valid = (indices >= 0) & (indices < count)
                if case.ratio == 4 and (query_group > 1 or dense_c4 or joint_sparse):
                    cmp_multiplicity = torch.bincount(indices[valid], minlength=(last_pos + 1) // 4).float()
                    indices = torch.arange(cmp_multiplicity.numel())
                    valid = cmp_multiplicity > 0
                elif case.ratio == 4 and count < indices.numel() <= 1024:
                    indices = indices[valid]
                    valid = torch.ones_like(indices, dtype=torch.bool)
                groups.append((case.cmp, case.cmp_block_table, indices, valid, cmp_quant))
            if (((joint_dense and case.ratio == 128) or (joint_sparse and case.ratio == 4))
                    and splits == 1 and (case.ratio == 128 or query_group == 1) and tile_size <= joint_tile_limit
                    and len(groups) == 2 and len(groups[1][2]) <= tile_size and len(original) <= tile_size
                    and (joint_sparse or len(groups[1][2]) > 0)):
                assert residual_quantization and ori_row_factors is None and cmp_row_factors is None
                caches = [gather(cache, table, batch, indices[valid])
                          for cache, table, indices, valid, quant in groups]
                biases = ([None, cmp_multiplicity[groups[1][3]].log()] if joint_sparse else None)
                out, lse = joint_quantized_row(case.q[t], caches, (ori_quant, cmp_quant),
                                             case.sink, case.softmax_scale, biases)
                result.append(out)
                if return_lse:
                    lses.append(lse)
                continue
            maximum = case.sink.repeat(splits, 1)
            denominator = torch.zeros(splits, case.q.shape[1])
            denominator[0].fill_(1)
            numerator = torch.zeros(splits, case.q.shape[1], DIM)
            tile_number = 0
            for kind, (cache, table, indices, valid, quant) in enumerate(groups):
                folded = case.q[t].float() * quant.scale
                alpha = (folded.abs().amax(-1) * (1.0 / 127)).clamp_min(1e-12)
                normalized_q = (folded * alpha.reciprocal()[:, None]).clamp(-127, 127)
                q8 = normalized_q.round().int()
                if residual_quantization:
                    q_low = ((normalized_q - q8.float()) * 127).round().int()
                for start in range(0, indices.numel(), tile_size):
                    part = tile_number % splits
                    tile_number += 1
                    ix, keep = indices[start:start + tile_size], valid[start:start + tile_size]
                    kv = torch.zeros((ix.numel(), DIM), dtype=torch.int32)
                    kv[keep] = (gather_slots(cache, ix[keep]) if kind == 0 and case.ori_indices is not None
                                else gather(cache, table, batch, ix[keep])).int()
                    dots = (q8 @ kv.T).float()
                    if residual_quantization:
                        dots += (q_low @ kv.T).float() * (1.0 / 127)
                    logits = dots * (alpha[:, None] * case.softmax_scale)
                    factor = None
                    row_factors = ori_row_factors if kind == 0 else cmp_row_factors
                    if row_factors is not None:
                        factor = torch.ones(ix.numel(), dtype=torch.float32)
                        block = cache.shape[1]
                        pages = (ix[keep] // block if kind == 0 and case.ori_indices is not None
                                 else table[batch, ix[keep] // block].long())
                        factor[keep] = row_factors[pages, ix[keep] % block]
                        logits *= factor[None, :]
                    if kind == 1 and cmp_multiplicity is not None:
                        logits += cmp_multiplicity[ix].clamp_min(1).log()
                    logits[:, ~keep] = -1e30
                    updated_max = torch.maximum(maximum[part], logits.amax(-1))
                    old_factor = torch.exp(maximum[part] - updated_max)
                    weights = torch.exp(logits - updated_max[:, None])
                    denominator[part] = denominator[part] * old_factor + weights.sum(-1)
                    if factor is not None:
                        weights *= factor[None, :]
                    p8 = (weights * 127).round().int()
                    dots = (p8 @ kv).float()
                    if residual_quantization:
                        p_low = ((weights * 127 - p8.float()) * 127).round().int()
                        dots += (p_low @ kv).float() * (1.0 / 127)
                    pv = dots * quant.scale * (1.0 / 127)
                    numerator[part] = numerator[part] * old_factor[:, None] + pv
                    maximum[part] = updated_max
            factors = torch.exp(maximum - maximum.amax(0))
            num = (numerator * factors[:, :, None]).sum(0)
            den = (denominator * factors).sum(0)
            result.append((num / den[:, None]).bfloat16())
            if return_lse:
                lses.append(maximum.amax(0) + den.log())
    return (torch.stack(result), torch.stack(lses)) if return_lse else torch.stack(result)


def make_case(*, query_lengths=(1,), seq_lens=(256,), heads=8, ratio=1,
              block_size=32, topk=1024, seed=0):
    """Deterministic varied paged inputs; every sequence has a shuffled page map."""
    gen = torch.Generator().manual_seed(seed)
    lengths = torch.tensor(seq_lens, dtype=torch.int32)
    cu_q = torch.tensor([0, *torch.tensor(query_lengths).cumsum(0).tolist()], dtype=torch.int32)
    q = torch.randn(int(cu_q[-1]), heads, DIM, generator=gen).to(torch.bfloat16)
    sink = torch.linspace(-2, 2, heads)

    def cache_for(lens):
        blocks = max(1, (max(lens) + block_size - 1) // block_size)
        pages = blocks * len(lens)
        table = torch.randperm(pages, generator=gen).reshape(len(lens), blocks).to(torch.int32)
        data = torch.randn(pages, block_size, 1, DIM, generator=gen).to(torch.bfloat16)
        return data, table

    ori, obt = cache_for(seq_lens)
    cmp, cbt, indices = None, None, None
    if ratio != 1:
        cmp, cbt = cache_for([length // ratio for length in seq_lens])
        if ratio == 4:
            indices = torch.full((q.shape[0], 1, topk), -1, dtype=torch.int32)
            for batch, length in enumerate(seq_lens):
                begin, end = int(cu_q[batch]), int(cu_q[batch + 1])
                for t in range(begin, end):
                    count = (length - (end - begin) + t - begin + 1) // ratio
                    ids = torch.randperm(count, generator=gen)[:topk]
                    indices[t, 0, :ids.numel()] = ids.to(torch.int32)
    return SharedKVCase(q, ori, obt, cu_q, lengths, sink, cmp, cbt, indices, ratio)
