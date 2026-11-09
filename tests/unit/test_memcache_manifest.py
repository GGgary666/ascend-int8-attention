import torch

from npu_sparse_attn_sharedkv_int8.memcache_manifest import scale_values_digest


def test_scale_identity_ignores_serialization_metadata_and_storage(tmp_path):
    values = {"layer.0": torch.linspace(0.01, 1.0, 512, dtype=torch.float32)}
    first, second = tmp_path / "first.pt", tmp_path / "second.pt"
    torch.save({"scales": values, "date": "first"}, first)
    torch.save({"scales": {key: value.clone() for key, value in values.items()}, "date": "second"}, second)
    assert first.read_bytes() != second.read_bytes()
    assert scale_values_digest(torch.load(first, weights_only=True)["scales"]) == \
        scale_values_digest(torch.load(second, weights_only=True)["scales"])
    changed = {key: value.clone() for key, value in values.items()}
    changed["layer.0"][0] += 0.01
    assert scale_values_digest(changed) != scale_values_digest(values)


def _installed_spec_hook(compact_pages, original):
    # Exercise the actual hook assigned to both model cache-spec classes,
    # without importing vLLM or initializing an NPU in the CPU contract suite.
    import ast
    from dataclasses import replace
    from pathlib import Path
    from npu_sparse_attn_sharedkv_int8.memcache_manifest import validate_transfer_mode

    source = Path(__file__).resolve().parents[2] / 'npu_sparse_attn_sharedkv_int8/vllm_adapter.py'
    install = next(node for node in ast.parse(source.read_text()).body
                   if isinstance(node, ast.FunctionDef) and node.name == 'install')
    hook = next(node for node in ast.walk(install)
                if isinstance(node, ast.FunctionDef) and node.name == 'spec')
    namespace = dict(torch=torch, replace=replace, compact_pages=compact_pages,
                     original_spec=original, validate_transfer_mode=validate_transfer_mode)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[hook], type_ignores=[])),
                 str(source), 'exec'), namespace)
    return namespace['spec']


def test_noncompact_transfer_rejected_before_upstream_cache_spec():
    from types import SimpleNamespace
    import pytest

    called = []
    hook = _installed_spec_hook(False, lambda *_: called.append(True))
    for connector in ('AscendStoreConnector', 'MultiConnector', 'MooncakeConnector'):
        config = SimpleNamespace(kv_transfer_config=SimpleNamespace(kv_connector=connector))
        with pytest.raises(ValueError, match='INT8 KV transfer requires compact pages'):
            hook(None, config)
    # Presence, not truthiness, controls the check.
    with pytest.raises(ValueError, match='INT8 KV transfer requires compact pages'):
        hook(None, SimpleNamespace(kv_transfer_config={}))
    assert called == []


def test_standalone_layouts_and_compact_transfer_keep_spec_contract():
    from dataclasses import dataclass
    from types import SimpleNamespace

    @dataclass
    class Spec:
        head_size: int = 512
        dtype: object = torch.bfloat16
        page_size_bytes: int = 8192
        page_size_padded: object = None
        indexes_kv_by_block_stride: bool = False

    for compact, transfer in ((False, None), (True, None), (True, object())):
        calls = []
        def original(*_):
            calls.append(True)
            return Spec()
        result = _installed_spec_hook(compact, original)(None, SimpleNamespace(kv_transfer_config=transfer))
        assert calls == [True]
        assert result.dtype == torch.int8
        assert result.page_size_padded == (None if compact else 8192)
        assert result.indexes_kv_by_block_stride
    assert _installed_spec_hook(False, lambda *_: None)(None, SimpleNamespace()) is None
