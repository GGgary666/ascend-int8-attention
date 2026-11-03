"""Opt-in integration. Unset SHAREDKV_INT8_MODE leaves vLLM unchanged."""

import os

_registered = False
_observer = None


def layer_prefix(impl):
    return impl.swa_cache_layer.prefix.removesuffix(".swa_cache")


def register():
    global _registered, _observer
    diagnostics = os.environ.get("SHAREDKV_PROFILE_DIAGNOSTICS_DIR")
    if diagnostics:
        from .serve_diagnostics import install as install_diagnostics
        install_diagnostics(diagnostics)
    mode = os.environ.get("SHAREDKV_INT8_MODE", "")
    if not mode or _registered:
        return
    if mode not in ("calibrate", "int8"):
        raise ValueError(f"Unsupported SHAREDKV_INT8_MODE={mode!r}")
    directory = os.environ.get("SHAREDKV_CALIBRATION_DIR")
    if mode == "calibrate" and not directory:
        raise ValueError("calibrate mode requires SHAREDKV_CALIBRATION_DIR")

    import vllm_ascend
    vllm_ascend._ensure_global_patch()
    from vllm_ascend.attention.dsa_v1 import AscendDSAImpl
    from vllm_ascend.models.deepseek_v4.dspark import DeepseekV4DSparkModel
    if mode == "calibrate":
        from .calibration import CacheObserver
        _observer = CacheObserver(directory)
        original_write = AscendDSAImpl._write_kv_cache

        def observed_write(self, cache, kv, slot_mapping, *, quantize):
            if self.turboquant is not None:
                raise ValueError("Shared-KV calibration requires the ordinary BF16 DSA path")
            _observer.observe(layer_prefix(self), "cmp" if quantize else "ori", cache, kv, slot_mapping)
            return original_write(self, cache, kv, slot_mapping, quantize=quantize)

        AscendDSAImpl._write_kv_cache = observed_write
    else:
        from .vllm_adapter import install
        install()

    # DSpark precomputes target-context KV outside AscendDSAImpl.forward.
    # Route that writer through the same observer; retain the existing slot
    # conversion and underlying BF16 scatter operation.
    def store_draft_context(self, shared_kv, slot_mapping, attn=None):
        if slot_mapping is None or slot_mapping.numel() == 0:
            return
        assert attn is not None
        layer = attn.dsa_attn.swa_cache_layer
        cache = getattr(layer, "kv_cache", None)
        if cache is None:
            return
        while isinstance(cache, (tuple, list)) and len(cache) == 1:
            cache = cache[0]
        impl = attn.dsa_attn.dsa_attn.impl
        if slot_mapping.ndim == 1:
            slot_mapping = impl.get_kv_plan().format_dsa_slot_mapping(slot_mapping, layer.block_size)
        impl._write_kv_cache(cache, shared_kv, slot_mapping, quantize=False)

    DeepseekV4DSparkModel._store_standard_swa_kv = store_draft_context
    _registered = True
