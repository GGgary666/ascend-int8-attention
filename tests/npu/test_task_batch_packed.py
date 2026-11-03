"""Independent tiny-K requests, packing limits, and macro-task tails."""
from tests.npu.test_capacity import main


def cases():
    rows = []
    # 24 cores: cover one through five requests/core, including mixed
    # packed/unpacked tails and more than one retained batch per core.
    for ratio in (4, 128):
        for batch in (25, 49, 73, 97, 121, 241):
            maximum = 3 if ratio == 4 else 6
            rows.append(dict(id=f'packed_c{ratio}_b{batch}_mixed{maximum}', ratio=ratio,
                queries=[1] * batch, lengths=[1 + i % maximum for i in range(batch)],
                capacity=(batch + 15) // 16 * 16, mode='graph', draft=False, index=batch % 3))
    # Cross every count*window<=32 boundary using unequal request lengths.
    # C128 remains a single KV kind throughout these tests.
    for batch, limits in ((25, (16, 17)), (49, (10, 11)),
                          (73, (8, 9)), (97, (6, 7))):
        for maximum in limits:
            rows.append(dict(id=f'packed_c128_b{batch}_limit{maximum}', ratio=128,
                queries=[1] * batch, lengths=[maximum if i % 3 else 1 for i in range(batch)],
                capacity=(batch + 15) // 16 * 16, mode='graph', draft=False, index=maximum % 3))
    return rows


if __name__ == '__main__':
    main(cases)
