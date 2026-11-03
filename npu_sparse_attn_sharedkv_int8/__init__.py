"""INT8 shared-KV attention. CPU references import without torch_npu or vLLM."""
from importlib import import_module

__version__ = "0.65.1"
__stage__ = "v65-p101-stage1"

__all__ = ["CacheQuantization", "SharedKVCase", "PreparedAttention", "PreparedCacheWrite"]

_EXPORTS = {
    "CacheQuantization": ("reference", "CacheQuantization"),
    "SharedKVCase": ("reference", "SharedKVCase"),
    "PreparedAttention": ("attention", "PreparedAttention"),
    "PreparedCacheWrite": ("cache", "PreparedCacheWrite"),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(f".{module}", __name__), attribute)
    globals()[name] = value
    return value
