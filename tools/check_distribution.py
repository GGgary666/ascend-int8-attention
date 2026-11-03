"""Install a wheel into a temporary target and exercise only its CPU reference."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


SMOKE = r'''
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
import torch
torch.set_num_threads(2)
from dataclasses import replace
import npu_sparse_attn_sharedkv_int8 as package
from npu_sparse_attn_sharedkv_int8 import CacheQuantization, SharedKVCase
from npu_sparse_attn_sharedkv_int8._paths import SCALES_PATH, NATIVE_DIR
from npu_sparse_attn_sharedkv_int8.reference import make_case, tiled_quantized_reference
from npu_sparse_attn_sharedkv_int8.vllm_plugin import register
assert Path(package.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve())
assert SCALES_PATH.is_file()
record = json.loads((NATIVE_DIR / 'manifest.json').read_text())
assert record['complete'] and len(record['libraries']) == 4
for name, expected in record['libraries'].items():
    assert hashlib.sha256((NATIVE_DIR / name).read_bytes()).hexdigest() == expected
entry = importlib.metadata.distribution('npu-sparse-attn-sharedkv-int8')
assert any(e.name == 'sharedkv_int8' and e.group == 'vllm.general_plugins' for e in entry.entry_points)
for ratio in (1, 4, 128):
    case = make_case(query_lengths=(1,), seq_lens=(129,), ratio=ratio)
    oq = CacheQuantization.calibrate([case.ori])
    cq = CacheQuantization.calibrate([case.cmp]) if case.cmp is not None else None
    encoded = replace(case, ori=oq.quantize(case.ori), cmp=cq.quantize(case.cmp) if cq else None)
    output = tiled_quantized_reference(encoded, oq, cq)
    assert output.shape == (1, 8, 512) and output.isfinite().all()
register()
assert 'torch_npu' not in sys.modules and 'vllm' not in sys.modules
print('PASS: isolated wheel, four native resources, calibration, plugin entry, R1/C4/C128 CPU references')
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    wheel = args.wheel.resolve()
    if not wheel.is_file() or wheel.suffix != ".whl":
        parser.error("Provide a wheel built after ./build.sh")
    env = {k: v for k, v in os.environ.items() if not k.startswith("SHAREDKV_")}
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    with tempfile.TemporaryDirectory(prefix="sharedkv-wheel-check-") as temporary:
        target = Path(temporary) / "installed"
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(target), str(wheel)],
            check=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        subprocess.run([sys.executable, "-I", "-c", SMOKE, str(target)],
                       cwd=temporary, env=env, check=True)


if __name__ == "__main__":
    main()
