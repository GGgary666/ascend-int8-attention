"""Write BF16 updates to fixed-scale INT8 pages on a caller-selected NPU."""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:1")
    args = parser.parse_args()
    import torch
    import torch_npu  # noqa: F401
    from npu_sparse_attn_sharedkv_int8 import CacheQuantization, PreparedCacheWrite

    torch.npu.set_device(args.device)
    values = torch.linspace(-1, 1, 4 * 512).reshape(4, 512).bfloat16()
    quantizer = CacheQuantization.calibrate([values])
    cache = torch.zeros((2, 32, 1, 512), dtype=torch.int8, device=args.device)
    slots = torch.tensor([0, 31, 32, 63], dtype=torch.int32, device=args.device)
    writer = PreparedCacheWrite(cache, values.to(args.device), slots,
                                quantizer.inverse_scale.to(args.device))
    with torch.inference_mode():
        writer.run()
        torch.npu.synchronize()
    actual = cache.reshape(-1, 512)[slots.long()].cpu()
    torch.testing.assert_close(actual, quantizer.quantize(values), rtol=0, atol=0)
    print("PASS: four exact INT8 slot writes across two pages")


if __name__ == "__main__":
    main()
