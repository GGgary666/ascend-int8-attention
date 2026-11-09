"""Run one foreground process from a generated deployment; the caller owns its lifetime."""

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import threading


def main(args):
    directory = args.deployment.resolve(strict=True)
    settings = json.loads((directory / "deployment.json").read_text())
    if args.role == "meta":
        os.environ["MMC_META_CONFIG_PATH"] = str(directory / "mmc-meta.conf")
        from memcache_hybrid import MetaService

        MetaService.main()
        return
    if args.role == "pool":
        os.environ["MMC_LOCAL_CONFIG_PATH"] = str(directory / "mmc-pool.conf")
        from memcache_hybrid import DistributedObjectStore

        stop = threading.Event()
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, lambda *_: stop.set())
        store = DistributedObjectStore()
        try:
            status = store.init(0)
            if status != 0:
                raise RuntimeError(f"MemCache pool initialization failed: {status}")
            print("MEMCACHE_POOL_READY", flush=True)
            stop.wait()
        finally:
            store.close()
        return
    environment = os.environ.copy()
    for name in list(environment):
        if name.startswith("SHAREDKV_") or name == "VLLM_ASCEND_EARLY_SHARED_MLP":
            del environment[name]
    environment.update(settings["environment"])
    for name, key in [
        ("PYTHONPATH", "pythonpath_prepend"),
        ("LD_LIBRARY_PATH", "librarypath_prepend"),
    ]:
        environment[name] = os.pathsep.join(settings[key] + [environment.get(name, "")])
    executable = Path(sys.executable).with_name("vllm")
    if not executable.is_file():
        raise FileNotFoundError("Install vLLM in the selected Python environment")
    cwd = Path(settings["working_directory"])
    cwd.mkdir(parents=True, exist_ok=True)
    os.chdir(cwd)
    os.execve(str(executable), [str(executable), *settings["command"]], environment)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("meta", "pool", "model"))
    parser.add_argument("--deployment", type=Path, required=True)
    main(parser.parse_args())
