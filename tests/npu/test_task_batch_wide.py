"""Retained KV capacity boundaries, including the next online tile.

Production encoding, page strides, padding, numerical gates and graph
timing are shared with test_capacity. The sizes exercise every new L0 K
tail, changing retained-task widths and heterogeneous request lengths.
"""
from tests.npu.test_capacity import main


def cases():
    rows = []
    for ratio in (4, 128):
        for i, keys in enumerate((256, 257, 289, 321, 353, 385, 417, 449, 481, 512, 513)):
            batch = (25, 49, 73)[i % 3]
            rows.append(dict(id=f'task_wide_c{ratio}_cmp{keys}', ratio=ratio,
                queries=[1] * batch, lengths=[keys * ratio + ratio - 1] * batch,
                capacity=(batch + 15) // 16 * 16, mode='graph', draft=False, index=i % 3))
        for capkeys in (320, 512):
            keys = (0, 1, 127, 128, 129, 256, 257, capkeys)
            rows.append(dict(id=f'task_wide_c{ratio}_mixed{capkeys}', ratio=ratio,
                queries=[1] * 71, lengths=[max(1, keys[i % len(keys)] * ratio) for i in range(71)],
                capacity=80, mode='graph', draft=False, index=1))
    return rows


if __name__ == '__main__':
    main(cases)
