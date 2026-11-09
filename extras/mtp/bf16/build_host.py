"""Build the isolated ordered launcher bridge; no NPU execution."""
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MAX_JOBS','2')
import torch_npu
from torch.utils.cpp_extension import load

npu=Path(torch_npu.__file__).resolve().parent
out=ROOT/'host_build';out.mkdir(exist_ok=True)
module=load(name='sharedkv_pointer_host',sources=[str(ROOT/'host_pointer.cpp')],
    extra_include_paths=[str(ROOT/'csrc/kernels/c8'),str(npu/'include'),
                         '/usr/local/Ascend/cann-9.2.0/include'],
    extra_cflags=['-O2','-std=c++17'],
    extra_ldflags=['-L'+str(npu/'lib'),'-ltorch_npu','-Wl,-rpath,'+str(npu/'lib'),
                   '-L/usr/local/Ascend/cann-9.2.0/lib64','-lascendcl'],
    build_directory=str(out),verbose=True)
print('Isolated ordered bridge:',module.__file__)
