"""Build ordered torch_npu bridges and stage all four package-local libraries."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=("all", "core", "fallback"), default="all")
    args = parser.parse_args()
    import torch_npu
    from torch.utils.cpp_extension import load

    root = Path(__file__).resolve().parents[1]
    npu = Path(torch_npu.__file__).resolve().parent
    cann = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann-9.2.0"))
    os.environ.setdefault("MAX_JOBS", os.environ.get("SHAREDKV_BUILD_JOBS", "2"))
    native = root / "npu_sparse_attn_sharedkv_int8" / "_native"
    native.mkdir(exist_ok=True)
    entries = (
        ("fallback", "sharedkv_host", "csrc/host/launch.cpp", "csrc/kernels", "sharedkv_int8"),
        ("core", "sharedkv_c8_host", "csrc/host/c8_launch.cpp", "csrc/kernels/c8", "sharedkv_c8"),
    )
    for component, name, source, include, library in entries:
        if args.component not in ("all", component):
            continue
        device = root / "build" / f"lib{library}.so"
        if not device.is_file():
            parser.error(f"Missing {device}; run ./build.sh --device-only first")
        build = root / "build" / "host" / component
        build.mkdir(parents=True, exist_ok=True)
        # The host bridge uses $ORIGIN both during load() and after packaging.
        shutil.copy2(device, build / device.name)
        inputs = [root / source, *sorted((root / include).glob("*.h"))]
        signature = {str(p.relative_to(root)): digest(p) for p in inputs}
        stamp = build / "source_signature.json"
        previous = json.loads(stamp.read_text()) if stamp.exists() else None
        if signature != previous:
            for obj in build.glob("*.o"):
                obj.unlink()
        load(
            name=name, sources=[str(root / source)], build_directory=str(build), verbose=True,
            extra_include_paths=[str(root / include), str(npu / "include"), str(cann / "include")],
            extra_cflags=["-O3"],
            extra_ldflags=[f"-L{npu}/lib", "-ltorch_npu", f"-L{root}/build", "-l" + library,
                           "-Wl,-rpath,'$$ORIGIN'", f"-Wl,-rpath,{npu}/lib"],
        )
        stamp.write_text(json.dumps(signature, indent=2) + "\n")
        for path in [device, build / f"{name}.so"]:
            temporary = native / (path.name + ".tmp")
            shutil.copy2(path, temporary)
            temporary.replace(native / path.name)

    required = ("libsharedkv_c8.so", "libsharedkv_int8.so", "sharedkv_host.so", "sharedkv_c8_host.so")
    inputs = [*sorted((root / "csrc").rglob("*.cpp")), *sorted((root / "csrc").rglob("*.h")),
              root / "CMakeLists.txt", Path(__file__).resolve()]
    manifest = {
        "release": "v65-p101-stage1",
        "source_sha256": {str(p.relative_to(root)): digest(p) for p in inputs},
        "libraries": {name: digest(native / name) for name in required if (native / name).is_file()},
        "complete": all((native / name).is_file() for name in required),
        "portability": "Rebuild for a different Python/torch_npu/CANN platform; device libraries use $ORIGIN.",
    }
    (native / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Staged {len(manifest['libraries'])} libraries in {native}")


if __name__ == "__main__":
    main()
