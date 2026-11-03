"""Independent-request batching: Cube K tails, L1 capacity and ragged lengths.

Uses the same real activation/KV encoding, unchanged numerical gates and
paired graph protocol as the capacity suite. No per-case production selector.
"""
from tests.npu.test_capacity import main


def cases():
    shapes = [
        ('no_compressed', [1] * 49, [127] * 49, 64),
        ('first_compressed', [1] * 97, [128] * 97, 128),
        ('macro_tail', [1] * 193, [129] * 193, 224),
        ('cmp33', [1] * 49, [4225] * 49, 64),
        ('cmp65', [1] * 73, [8321] * 73, 80),
        ('cmp97', [1] * 97, [12417] * 97, 128),
        ('cmp129', [1] * 49, [16513] * 49, 64),
        ('cmp161', [1] * 73, [20609] * 73, 80),
        ('cmp193', [1] * 97, [24705] * 97, 128),
        ('cmp225', [1] * 97, [28801] * 97, 128),
        ('mixed_short', [1] * 71, [[1, 63, 127, 128, 129, 513, 8193, 12417][i % 8] for i in range(71)], 80),
        ('mixed_medium', [1] * 71, [[127, 129, 4225, 8321, 16513, 20609, 24705, 32767][i % 8] for i in range(71)], 96),
        ('q3_tail', [3] * 19, [129] * 19, 64),
        ('q5_tail', [5] * 29, [1028] * 29, 160),
        ('q6_short', [6] * 41, [513] * 41, 256),
    ]
    rows = [dict(id='task_batch_' + name, ratio=128, queries=q, lengths=s,
                 capacity=cap, mode='graph', draft=False, index=i % 3)
            for i, (name, q, s, cap) in enumerate(shapes)]
    sparse = [
        ('no_compressed', 49, [3] * 49, 64),
        ('first_compressed', 97, [4] * 97, 128),
        ('cmp33', 49, [133] * 49, 64),
        ('cmp65', 73, [261] * 73, 80),
        ('cmp97', 97, [389] * 97, 128),
        ('cmp129', 73, [517] * 73, 80),
        ('mixed_short', 71, [[1, 3, 4, 127, 129, 256, 511, 513][i % 8] for i in range(71)], 80),
        ('mixed_medium', 71, [[1, 128, 129, 511, 513, 517, 768, 1024][i % 8] for i in range(71)], 80),
    ]
    rows += [dict(id='task_batch_c4_' + name, ratio=4, queries=[1] * batch,
                  lengths=lengths, capacity=cap, mode='graph', draft=False, index=i % 3)
             for i, (name, batch, lengths, cap) in enumerate(sparse)]
    return rows


if __name__ == '__main__':
    main(cases)
