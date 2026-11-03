"""Production cache views and independent integer-oracle geometry for tests."""
from dataclasses import replace
import torch
from npu_sparse_attn_sharedkv_int8.attention import PreparedAttention
from npu_sparse_attn_sharedkv_int8.cache import PreparedCacheWrite
from tests.support.serving_shapes import Geometry, INT8_PAGE_STRIDE


def metrics(actual, expected):
    a, b = actual.double(), expected.double()
    flat_a, flat_b = a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
    cosine = torch.nn.functional.cosine_similarity(flat_a, flat_b, dim=-1)
    return dict(relative_l2=float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b)),
                min_head_cosine=float(cosine.min()))


def prepare(corpus, c, device):
    geometry = Geometry(c['id'], c['ratio'], tuple(c['queries']), tuple(c['lengths']),
                        padded_tokens=c['capacity'], draft_slots=c.get('draft', False),
                        graph=c['mode'] == 'graph')
    source, oq, cq, provenance = corpus.make(geometry, layer_index=c.get('index', 0))
    quant = replace(source, ori=oq.quantize(source.ori), cmp=cq.quantize(source.cmp) if cq else None)
    dev = quant.to(device)
    for kind, scale in (('ori', oq), ('cmp', cq)):
        if scale is None:
            continue
        original = getattr(source, kind)
        raw = torch.empty((original.shape[0], INT8_PAGE_STRIDE), dtype=torch.int8, device=device)
        view = raw.as_strided(original.shape, (INT8_PAGE_STRIDE, 512, 512, 1))
        updates = original.reshape(-1, 1, 512).to(device)
        slots = torch.arange(len(updates), dtype=torch.int32, device=device)
        PreparedCacheWrite(view, updates, slots, scale.inverse_scale.to(device)).run()
        assert torch.equal(view.cpu(), getattr(quant, kind)), 'Production writer encoding changed'
        setattr(dev, kind, view)
    dev = pad_case(dev, c['capacity'], geometry.graph)
    out_guard = torch.full((c['capacity']+2, 8, 512), 123, dtype=torch.bfloat16, device=device)
    lse_guard = torch.full((c['capacity']+2, 8), 123, dtype=torch.float32, device=device)
    entry = PreparedAttention(dev, oq.scale.to(device), cq.scale.to(device) if cq else None,
        output=out_guard[1:-1], return_lse=True, lse_output=lse_guard[1:-1])
    workspace_guard = torch.full((entry.workspace.numel()+1024,), 0xA5, dtype=torch.uint8, device=device)
    entry.workspace = workspace_guard[512:-512]
    entry.args.workspace = entry.workspace.data_ptr()
    return dict(source=source, quant=quant, dev=dev, original=entry, entries={'latest': entry},
                live=sum(c['queries']), guards=(out_guard, lse_guard, workspace_guard),
                quantizers=(oq, cq), provenance=provenance)


def guards(state):
    out, lse, workspace = state['guards']
    assert (out[[0, -1]].cpu() == 123).all() and (lse[[0, -1]].cpu() == 123).all()
    assert (workspace[:512].cpu() == 0xA5).all() and (workspace[-512:].cpu() == 0xA5).all()
    entry = state['original']
    assert (entry.out[state['live']:].cpu() == 0).all()
    assert (entry.lse[state['live']:].cpu() == 0).all()

def layout(e,c):
    if not getattr(e,'c8_dynamic',False):return e.tile_size,e.query_group,e.splits
    q=sum(c['queries']);keys=max(c['lengths'])//c['ratio'];pairs=sum((v+1)//2 for v in c['queries'])
    packed=[]
    for count,length in zip(c['queries'],c['lengths']):
        packed.extend((3 if min(4,count-start)>2 else 2)*(1+(length//128+255)//256)
                      for start in range(0,count,4))
    packed_load=[0]*24
    for task,cost in enumerate(packed):
        packed_load[23-task%24 if (task//24)%2 else task%24]+=cost
    packed_peak=min(max(packed_load), max(sum(packed[core::24]) for core in range(24)))
    if (q>48 and 24<len(packed)<=48 and len(packed)<pairs and
            len({(length//128+255)//256 for length in c['lengths']})>1):
        ordered=sorted(packed,reverse=True)
        packed_peak=min(packed_peak,max(ordered[0],ordered[48-len(packed)]+ordered[24]))
    if c['ratio']==4:
        n,g=(((128 if keys<=128 else 256 if keys<=256 else 512),1) if
             ((keys<=128 and q<24) or (keys<=960 and q>=24 and all(v<=1 for v in c['queries']))) else
             (256,2) if q>=24 and keys<=256 else (512,1))
    elif q>48:
        wide=[];paired=[]
        for count,length in zip(c['queries'],c['lengths']):
            wide.extend([1+(length//128+511)//512]*count)
            paired.extend([1+(length//128+255)//256]*((count+1)//2))
        peak=lambda costs:max(sum(costs[core::24]) for core in range(24))
        if len(packed)<len(paired) and packed_peak<2*min(peak(paired),peak(wide)):n,g=256,4
        elif len(paired)<q and peak(paired)<peak(wide):n,g=256,2
        else:n,g=(128 if keys<=128 else 256 if keys<=256 else 512),1
    elif keys>256:
        costs=[1+(s//128+511)//512 for length,s in zip(c['queries'],c['lengths']) for _ in range(length)]
        wide=max(costs)
        if q>24:
            best=max(wide,max(costs[:q-24])+max(costs[24:]))
            ordered=sorted(costs,reverse=True)
            bound=max(wide,ordered[48-q]+ordered[24])
            best=min(best,bound)
            wide=best
        paired=(1+(keys+255)//256)*((pairs+23)//24)
        n,g=(256,2) if q>24 and paired<wide else (512,1)
    elif pairs<=24 and (q>24 or q>=4*len(c['queries'])):n,g=256,2
    else:n,g=(128 if keys<=128 else 256),1
    if c['ratio']==128 and 24<q<=48:
        costs=[1+(length//128+511)//512 for count,length in zip(c['queries'],c['lengths']) for _ in range(count)]
        wide=max(max(costs),max(costs[:q-24])+max(costs[24:]))
        if keys>256:
            order=sorted(costs,reverse=True)
            wide=min(wide,max(max(costs),order[48-q]+order[24]))
        pair_cost=(1+(keys+255)//256)*((pairs+23)//24)
        if len(packed)<pairs and packed_peak<2*min(wide,pair_cost):n,g=256,4
    s=1
    if (q>24 or (c['ratio']==4 and q==24)) and pairs==q and 512<keys<=960:
        n,g=(768 if keys<=768 else 896 if keys<=896 else 960),1
    if n==512 and g==1 and 0<q<=12:
        tiles=1+((min(keys,1024) if c['ratio']==4 else keys)+511)//512
        s=min(24//q,4,tiles)
    return n,g,s


def pad_case(case, capacity, graph):
    live = case.q.shape[0]
    q = torch.full((capacity,*case.q.shape[1:]), float('nan'), dtype=case.q.dtype, device=case.q.device)
    q[:live].copy_(case.q)
    changes = dict(q=q)
    for name in ('cmp_indices','ori_indices'):
        source = getattr(case,name)
        if source is not None:
            value = torch.full((capacity,*source.shape[1:]),-1,dtype=source.dtype,device=source.device)
            value[:live].copy_(source);changes[name]=value
    if graph and capacity>live:
        # DEFAULT target decode pads uniform q6 requests. Nonuniform stress
        # cases use the runner's other form: one dummy request ends at capacity.
        # Ref model_runner_v1.py::_pad_query_start_loc_for_fia (1092/1102).
        ends = (torch.arange(live+6,capacity+1,6,dtype=torch.int32,device=q.device)
                if live % 6 == 0 and capacity % 6 == 0 else
                torch.tensor([capacity],dtype=torch.int32,device=q.device))
        extra = len(ends)
        changes['cu_q'] = torch.cat((case.cu_q,ends))
        changes['seq_lens'] = torch.cat((case.seq_lens,torch.zeros(extra,dtype=torch.int32,device=q.device)))
        for name in ('ori_block_table','cmp_block_table'):
            source=getattr(case,name)
            if source is not None:changes[name]=torch.cat((source,torch.zeros((extra,source.shape[1]),dtype=source.dtype,device=q.device)))
    return replace(case,**changes)

