"""Load only the maintained attention/cache libraries and ordered host bridge."""
import ctypes
from functools import lru_cache
import importlib.util
import os
from types import SimpleNamespace

from ._paths import NATIVE_DIR, native_path



@lru_cache(maxsize=1)
def native_launcher():
    if os.environ.get("SHAREDKV_USE_CTYPES") == "1":
        return None
    path = NATIVE_DIR / "sharedkv_host.so"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("sharedkv_host", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "stream_id"):
        raise RuntimeError("Native host launcher is outdated; run ./build.sh")
    if os.environ.get("SHAREDKV_HOST_QUEUE") == "1":
        return SimpleNamespace(
            attention=module.attention_queued, cache_write=module.cache_write_queued,
            attention_args_size=module.attention_args_size, cache_args_size=module.cache_args_size,
            attention_variant_count=module.attention_variant_count, stream_id=module.stream_id,
            _owner=module,
        )
    return module


@lru_cache(maxsize=1)
def _library():
    return ctypes.CDLL(str(native_path("libsharedkv_int8.so")))


class Kernels:
    """Shared fallback/cache handle; no probe or experimental entry loading."""
    def __init__(self):
        self.lib = _library()
