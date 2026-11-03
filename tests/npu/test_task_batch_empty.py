"""One/two KV-kind boundaries with live/padded capacities and graph replay."""
from tests.npu.test_capacity import main


def cases():
    rows = []
    for ratio in (4, 128):
        for batch in (24, 25, 49, 97, 512):
            for length in (1, ratio - 1, ratio):
                rows.append(dict(id=f'task_empty_c{ratio}_b{batch}_s{length}', ratio=ratio,
                    queries=[1] * batch, lengths=[length] * batch,
                    capacity=(batch + 15) // 16 * 16, mode='graph', draft=False, index=0))
    return rows


if __name__ == '__main__':
    main(cases)
