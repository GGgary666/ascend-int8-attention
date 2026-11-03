"""Dense retained-KV resource limits and online continuation, independent shapes."""
from tests.npu.test_capacity import main


def ratio_cases(ratio):
    rows = []
    for i, keys in enumerate((511, 512, 513, 543, 544, 545, 640, 641, 672, 673,
                               704, 705, 736, 737, 767, 768, 769, 896, 897)):
        batch = (25, 49, 97)[i % 3]
        rows.append(dict(id=f'retained_c{ratio}_b{batch}_cmp{keys}', ratio=ratio,
            queries=[1]*batch, lengths=[ratio*keys+ratio-1]*batch,
            capacity=(batch+15)//16*16, mode='graph', draft=False, index=i%3))
    for batch in (31, 73, 129):
        for keys in (544, 768, 769, 800, 896, 897):
            extents = (1, 127, ratio*127, ratio*512, ratio*keys+ratio-1)
            rows.append(dict(id=f'retained_c{ratio}_b{batch}_mixed{keys}', ratio=ratio,
                queries=[1]*batch, lengths=[extents[i%len(extents)] for i in range(batch)],
                capacity=(batch+15)//16*16, mode='graph', draft=False, index=1))
    # Append new boundaries so the original case IDs and activation layers
    # remain stable for comparisons with frozen binaries.
    for i, keys in enumerate((800, 831, 832, 833, 863, 864, 865, 895)):
        batch = (33, 65, 101)[i % 3]
        rows.append(dict(id=f'retained_c{ratio}_b{batch}_cmp{keys}', ratio=ratio,
            queries=[1]*batch, lengths=[ratio*keys+ratio-1]*batch,
            capacity=(batch+15)//16*16, mode='graph', draft=False, index=i%3))
    for i, keys in enumerate((927,928,929,959,960,961)):
        batch=(25,49,97)[i%3]
        rows.append(dict(id=f'retained_c{ratio}_b{batch}_cmp{keys}',ratio=ratio,
            queries=[1]*batch,lengths=[ratio*keys+ratio-1]*batch,
            capacity=(batch+15)//16*16,mode='graph',draft=False,index=i%3))
    for batch in (31,73,129):
        for keys in (928,960,961):
            extents=(1,127,ratio*127,ratio*512,ratio*keys+ratio-1)
            rows.append(dict(id=f'retained_c{ratio}_b{batch}_mixed{keys}',ratio=ratio,
                queries=[1]*batch,lengths=[extents[i%len(extents)] for i in range(batch)],
                capacity=(batch+15)//16*16,mode='graph',draft=False,index=1))
    return rows


def cases():
    return ratio_cases(4) + ratio_cases(128)


if __name__ == '__main__':
    main(cases)
