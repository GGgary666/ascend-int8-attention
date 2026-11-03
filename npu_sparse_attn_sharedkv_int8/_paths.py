"""Package-local resources; independent of checkout name and working directory."""
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
NATIVE_DIR = PACKAGE_ROOT / "_native"
SCALES_PATH = PACKAGE_ROOT / "data" / "scales_v1.pt"


def native_path(filename):
    path = NATIVE_DIR / filename
    if not path.is_file():
        raise RuntimeError(
            f"Missing native library {filename}. Run ./build.sh in the source checkout "
            "before installing the NPU package."
        )
    return path
