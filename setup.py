"""Tag wheels containing the optional native bridges for this Python/platform."""
from pathlib import Path

from setuptools import Distribution, setup


class NativeDistribution(Distribution):
    def has_ext_modules(self):
        native = Path(__file__).parent / "npu_sparse_attn_sharedkv_int8" / "_native"
        return any(native.glob("*.so"))


setup(distclass=NativeDistribution)
