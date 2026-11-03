"""Default opt-in removal, explicit rollback and ABI boundary checks."""
import os
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch_npu  # noqa: F401
from npu_sparse_attn_sharedkv_int8.attention import PreparedAttention, _Args
from npu_sparse_attn_sharedkv_int8 import c8_attention
from npu_sparse_attn_sharedkv_int8.reference import make_case, CacheQuantization


def main():
    with patch.dict(os.environ, {}, clear=True):
        assert c8_attention.enabled()
    for flag, enabled in [('0', False), ('1', True)]:
        with patch.dict(os.environ, SHAREDKV_C8_DECODE=flag):
            assert c8_attention.enabled() is enabled
    with patch.dict(os.environ, SHAREDKV_C8_DECODE='typo'):
        try:
            c8_attention.enabled()
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid default selector silently accepted')
    valid = dict(heads=8, tokens=48, batch=8, ratio=4, window=128,
                 oriBlock=32, cmpBlock=32, oriIndices=0, topk=1024)
    assert c8_attention.supported(SimpleNamespace(blocks=24, args=_Args(**valid)))
    for key, value in [('tokens', 0), ('batch', 0), ('heads', 16), ('ratio', 1),
                       ('window', 129), ('oriBlock', 64), ('cmpBlock', 64),
                       ('oriIndices', 1), ('topk', 512)]:
        assert not c8_attention.supported(SimpleNamespace(blocks=24, args=_Args(**(valid | {key: value}))))
    for batch, tokens in [(9,49), (23,138), (128,768), (1024,8192)]:
        assert c8_attention.supported(SimpleNamespace(blocks=24, args=_Args(**(valid | dict(batch=batch,tokens=tokens)))))

    torch.set_num_threads(4); torch.npu.set_device(1)
    case = make_case(query_lengths=(1,), seq_lens=(129,), heads=8, ratio=4, topk=1024)
    oq, cq = CacheQuantization.calibrate([case.ori]), CacheQuantization.calibrate([case.cmp])
    from dataclasses import replace
    dev = replace(case, ori=oq.quantize(case.ori), cmp=cq.quantize(case.cmp)).to('npu:1')
    scales = oq.scale.to('npu:1'), cq.scale.to('npu:1')
    with patch.dict(os.environ, SHAREDKV_C8_DECODE='0'):
        rollback = PreparedAttention(dev, *scales)
        forced = PreparedAttention(dev, *scales, implementation='v65')
    assert not getattr(rollback, 'c8_dynamic', False) and forced.c8_dynamic
    explicit = PreparedAttention(dev, *scales, implementation='fallback')
    rollback.run(); expected = rollback.out.clone(); explicit.run()
    assert torch.equal(explicit.out.cpu(), expected.cpu())
    forced.run(); assert forced.out.cpu().isfinite().all()
    print('PASS: unset/0/1/invalid selector, ABI boundaries, unrestricted batch/query admission, explicit V65 and bitexact rollback')


if __name__ == '__main__':
    main()
