"""Live original-window tile boundaries, strided caches and padded graphs."""
from tests.npu.test_capacity import main


def cases():
    rows = []
    lengths = [31, 32, 33, 63, 64, 65, 95, 96, 97, 127, 128, 129]
    for ratio in (4, 128):
        for i, length in enumerate(lengths):
            batch = [25, 49, 97][i % 3]
            rows.append(dict(id=f'task_window_c{ratio}_s{length}', ratio=ratio,
                queries=[1] * batch, lengths=[length] * batch,
                capacity=(batch + 15) // 16 * 16, mode='graph', draft=False, index=i % 3))
        for maximum in (33, 65, 97, 129):
            seq = [1, maximum // 2, maximum]
            rows.append(dict(id=f'task_window_c{ratio}_mixed{maximum}', ratio=ratio,
                queries=[1] * 71, lengths=[seq[i % 3] for i in range(71)],
                capacity=80, mode='graph', draft=False, index=0))
    return rows


if __name__ == '__main__':
    main(cases)
