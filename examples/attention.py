"""Minimal NPU attention. Run only on an available, validated DAV2201 device."""
import argparse
from dataclasses import replace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:1")
    parser.add_argument("--ratio", type=int, choices=(4, 128), default=4)
    args = parser.parse_args()
    import torch
    import torch_npu  # noqa: F401
    from npu_sparse_attn_sharedkv_int8 import CacheQuantization, PreparedAttention
    from npu_sparse_attn_sharedkv_int8.reference import make_case

    torch.npu.set_device(args.device)
    case = make_case(query_lengths=(1,), seq_lens=(1024,), ratio=args.ratio)
    oq = CacheQuantization.calibrate([case.ori])
    cq = CacheQuantization.calibrate([case.cmp])
    encoded = replace(case, ori=oq.quantize(case.ori), cmp=cq.quantize(case.cmp)).to(args.device)
    operation = PreparedAttention(encoded, oq.scale.to(args.device), cq.scale.to(args.device))
    with torch.inference_mode():
        output = operation.run()
        torch.npu.synchronize()
    print({"shape": list(output.shape), "dtype": str(output.dtype), "device": str(output.device)})


if __name__ == "__main__":
    main()
