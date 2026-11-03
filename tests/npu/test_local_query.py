"""Independent-query ownership, mixed per-core extents and unseen capacities."""
from tests.npu.test_capacity import main


def cases():
    rows = []
    for ratio in (4, 128):
        for batch, length in ((65, 11), (129, 81), (241, 129), (513, 33), (1025, 127)):
            rows.append(dict(id=f'local_c{ratio}_b{batch}_s{length}', ratio=ratio,
                queries=[1]*batch, lengths=[length]*batch, capacity=(batch+15)//16*16,
                mode='graph', draft=False, index=0))
        for batch in (49, 97, 513):
            extents = (1, ratio*128, ratio*512, ratio*513+ratio-1)
            for mixed_within in (False, True):
                lengths = [extents[((i//24+i%7) if mixed_within else i%24)%4] for i in range(batch)]
                kind = 'within' if mixed_within else 'across'
                rows.append(dict(id=f'local_c{ratio}_b{batch}_mixed_{kind}', ratio=ratio,
                    queries=[1]*batch, lengths=lengths, capacity=(batch+15)//16*16,
                    mode='graph', draft=False, index=1))
    return rows


if __name__ == '__main__':
    main(cases)
