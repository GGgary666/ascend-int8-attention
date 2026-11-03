"""Run the frozen INT8 KV / Q-P residual protocol entirely on CPU."""
import argparse
from dataclasses import replace
import json

import torch

from npu_sparse_attn_sharedkv_int8 import CacheQuantization
from npu_sparse_attn_sharedkv_int8.reference import (
    make_case, sharedkv_reference, tiled_quantized_reference,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratio", type=int, choices=(1, 4, 128), default=4)
    args = parser.parse_args()
    torch.set_num_threads(2)
    case = make_case(query_lengths=(1,), seq_lens=(257,), ratio=args.ratio, seed=7)
    oq = CacheQuantization.calibrate([case.ori])
    cq = CacheQuantization.calibrate([case.cmp]) if case.cmp is not None else None
    encoded = replace(case, ori=oq.quantize(case.ori), cmp=cq.quantize(case.cmp) if cq else None)
    output = tiled_quantized_reference(encoded, oq, cq)
    golden, _ = sharedkv_reference(case)
    relative_l2 = float((output.float() - golden).norm() / golden.norm())
    print(json.dumps({"backend": "cpu integer reference", "ratio": args.ratio,
                      "output_shape": list(output.shape), "relative_l2_vs_bf16": relative_l2}, indent=2))


if __name__ == "__main__":
    main()
