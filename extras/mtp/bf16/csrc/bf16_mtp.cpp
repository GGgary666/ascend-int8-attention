// A3/DAV2201: BF16 MTP with the same ordered physical-slot reuse contract.
// Q/KV/P BF16; QK/PV Cube FP32 accumulation; softmax/normalization FP32.
#include "kernel_operator.h"
#include "lib/activation/softmaxflashv2.h"
#include "args.h"
using namespace AscendC;
namespace {
constexpr uint32_t M=16,N=256,D=512,R=8,G=QUERY_GROUP;
constexpr uint32_t SCORE=0,P=SCORE+M*N*4,PV=P+M*N*2,KV=PV+M*D*4;
constexpr uint32_t WORKSPACE=KV+N*D*2;
constexpr uint16_t KV_READY=5,SCORE_READY=7,P_READY=8,PV_READY=9;
constexpr IsResetLoad3dConfig LOAD_CFG{true,true};
constexpr SoftmaxConfig SM_CFG{false,0,0,SoftmaxMode::SOFTMAX_OUTPUT_WITHOUT_BRC};
static_assert(M*(D+N)*2+N*D*2<=512*1024-128);
template<class T> __aicore__ inline GlobalTensor<T> Gm(uint64_t address) {
    GlobalTensor<T> t;t.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(address));return t;
}
struct Selection {
    uint32_t token,rows;
    __aicore__ Selection(const SharedKVArgs& p,uint32_t task) {
        auto cu=Gm<int32_t>(p.cuQ);uint32_t begin=0,end=0;
        for(uint32_t b=0;b<p.batch;++b) {
            end=cu.GetValue(b+1);uint32_t groups=(end-begin+G-1)/G;
            if(task<groups)break;task-=groups;begin=end;
        }
        token=begin+task*G;uint32_t count=end-token;rows=(count<G?count:G)*8;
    }
};
__aicore__ uint32_t Tasks(const SharedKVArgs& p) {
    auto cu=Gm<int32_t>(p.cuQ);uint32_t begin=0,result=0;
    for(uint32_t b=0;b<p.batch;++b) {uint32_t end=cu.GetValue(b+1);result+=(end-begin+G-1)/G;begin=end;}
    return result;
}
__aicore__ void LoadA(LocalTensor<bfloat16_t> a0,LocalTensor<bfloat16_t> a1,uint32_t k) {
    LoadData3DParamsV2<bfloat16_t> x{};x.l1H=M/16;x.l1W=16;x.channelSize=k;x.padList[3]=255;
    x.mExtension=M;x.kExtension=k;x.strideW=x.strideH=x.filterW=x.filterH=1;
    x.dilationFilterW=x.dilationFilterH=1;LoadData<bfloat16_t,LOAD_CFG>(a0,a1,x);
}
__aicore__ void Store(GlobalTensor<float> dst,LocalTensor<float> src,uint32_t rows,uint32_t cols) {
    SetFlag<HardEvent::M_FIX>(EVENT_ID0);WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
    DataCopyCO12DstParams cp{};cp.mSize=rows;cp.nSize=cols;cp.srcStride=M;cp.dstStride=cols;
    cp.quantPre=QuantMode_t::NoQuant;cp.nz2ndEn=true;SetFixpipeNz2ndFlag(1,1,1);DataCopy(dst,src,cp);
}
__aicore__ void Cube(const SharedKVArgs& p) {
    TPipe pipe;TBuf<TPosition::A1> qb,pb;TBuf<TPosition::B1> kb;
    TBuf<TPosition::A2> ab;TBuf<TPosition::B2> bb;TBuf<TPosition::CO1> cb;
    pipe.InitBuffer(qb,M*D*2);pipe.InitBuffer(pb,M*N*2);pipe.InitBuffer(kb,N*D*2);
    pipe.InitBuffer(ab,M*D*2);pipe.InitBuffer(bb,65536);pipe.InitBuffer(cb,M*D*4);
    auto q1=qb.Get<bfloat16_t>(),p1=pb.Get<bfloat16_t>(),k1=kb.Get<bfloat16_t>();
    auto a0=ab.Get<bfloat16_t>(),b0=bb.Get<bfloat16_t>();auto c=cb.Get<float>();
    uint64_t ws=p.workspace+static_cast<uint64_t>(GetBlockIdx())*WORKSPACE;
    uint32_t tasks=Tasks(p);
    for(uint32_t task=GetBlockIdx();task<tasks;task+=GetBlockNum()) {
        Selection s(p,task);Nd2NzParams q{};q.ndNum=1;q.nValue=s.rows;q.dValue=D;q.srcDValue=D;q.dstNzC0Stride=M;q.dstNzNStride=1;
        DataCopy(q1,Gm<bfloat16_t>(p.q)[static_cast<uint64_t>(s.token)*8*D],q);
        CrossCoreWaitFlag(KV_READY);
        Nd2NzParams kv=q;kv.nValue=N;kv.dstNzC0Stride=N;DataCopy(k1,Gm<bfloat16_t>(ws+KV),kv);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        MmadParams mm{};mm.m=M;mm.n=N;mm.k=64;
        for(uint32_t k=0;k<D;k+=64) {
            LoadA(a0,q1[k*M],64);LoadData2DParams load{};load.srcStride=1;load.repeatTimes=(N/16)*(64/16);
            LoadData(b0,k1[k*N],load);
            SetFlag<HardEvent::MTE1_M>(EVENT_ID0);WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
            mm.cmatrixInitVal=k==0;Mmad(c,a0,b0,mm);PipeBarrier<PIPE_ALL>();
        }
        Store(Gm<float>(ws+SCORE),c,s.rows,N);CrossCoreSetFlag<2,PIPE_FIX>(SCORE_READY);
        CrossCoreWaitFlag(P_READY);PipeBarrier<PIPE_ALL>();
        Nd2NzParams pp=q;pp.dValue=N;pp.srcDValue=N;DataCopy(p1,Gm<bfloat16_t>(ws+P),pp);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        mm.n=D;mm.k=64;
        for(uint32_t k=0;k<N;k+=64) {
            LoadA(a0,p1[k*M],64);
            LoadData2dTransposeParams load{};load.repeatTimes=64/16;load.srcStride=1;load.dstGap=D/16-1;
            for(uint32_t n=0;n<D/16;++n)LoadDataWithTranspose(b0[n*256],k1[n*N*16+k*16],load);
            SetFlag<HardEvent::MTE1_M>(EVENT_ID0);WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
            mm.cmatrixInitVal=k==0;Mmad(c,a0,b0,mm);PipeBarrier<PIPE_ALL>();
        }
        Store(Gm<float>(ws+PV),c,s.rows,D);PipeBarrier<PIPE_ALL>();CrossCoreSetFlag<2,PIPE_FIX>(PV_READY);
    }
}
__aicore__ void Gather(const SharedKVArgs& p,const Selection& s,uint32_t sub,LocalTensor<bfloat16_t> kv,
    LocalTensor<int32_t> ix,LocalTensor<float> mask,LocalTensor<float> tmp,LocalTensor<uint8_t> bits,GlobalTensor<bfloat16_t> dst) {
    DataCopy(ix,Gm<int32_t>(p.oriIndices)[static_cast<uint64_t>(s.token)*N],N);
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    const uint32_t capacity=p.oriPages*p.oriBlock;
    auto clipped=tmp.ReinterpretCast<int32_t>();
    Maxs(clipped,ix,0,N);PipeBarrier<PIPE_V>();Mins(clipped,clipped,static_cast<int32_t>(capacity-1),N);PipeBarrier<PIPE_V>();
    Sub(clipped,ix,clipped,N);PipeBarrier<PIPE_V>();Cast(tmp,clipped,RoundMode::CAST_RINT,N);PipeBarrier<PIPE_V>();
    CompareScalar(bits,tmp,0.0f,CMPMODE::EQ,N);PipeBarrier<PIPE_V>();Duplicate(mask,0.0f,N);PipeBarrier<PIPE_V>();
    Select(mask,bits,mask,-1e30f,SELMODE::VSEL_TENSOR_SCALAR_MODE,N);
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    for(uint32_t chunk=0;chunk<N/2;chunk+=32) {
        Duplicate(kv.ReinterpretCast<int32_t>(),0,32*D/2);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        for(uint32_t n=0;n<32;) {
            uint32_t slot=sub*(N/2)+chunk+n;int32_t idx=ix.GetValue(slot);
            if(idx<0 || static_cast<uint32_t>(idx)>=capacity){++n;continue;}
            uint32_t run=1,limit=p.oriBlock-static_cast<uint32_t>(idx)%p.oriBlock;limit=limit<32-n?limit:32-n;
            while(run<limit && ix.GetValue(slot+run)==idx+static_cast<int32_t>(run))++run;
            uint64_t offset=static_cast<uint64_t>(static_cast<uint32_t>(idx)/p.oriBlock)*p.oriStride+static_cast<uint64_t>(static_cast<uint32_t>(idx)%p.oriBlock)*D;
            DataCopyPad(kv[n*D],Gm<bfloat16_t>(p.ori)[offset],DataCopyExtParams{1,run*D*2,0,0,0},DataCopyPadExtParams<bfloat16_t>{});n+=run;
        }
        SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        DataCopy(dst[(sub*(N/2)+chunk)*D],kv,32*D);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
    CrossCoreSetFlag<2,PIPE_MTE3>(KV_READY);
}
__aicore__ void Vector(const SharedKVArgs& p) {
    // 32KiB gather +16KiB FP32 score/output +8KiB BF16 P/output +32KiB softmax,
    // mask/index/small reduction buffers: below184KiB A3 user UB.
    TPipe pipe;TBuf<TPosition::VECCALC> gb,xb,hb,ixb,maskb,tmpb,bitsb,mb,sb,eb,sm,bcast,rb;
    pipe.InitBuffer(gb,32*D*2);pipe.InitBuffer(xb,R*D*4);pipe.InitBuffer(hb,R*D*2);
    pipe.InitBuffer(ixb,N*4);pipe.InitBuffer(maskb,N*4);pipe.InitBuffer(tmpb,N*4);pipe.InitBuffer(bitsb,256);
    pipe.InitBuffer(mb,2*R*4);pipe.InitBuffer(sb,2*R*4);pipe.InitBuffer(eb,R*4);
    pipe.InitBuffer(sm,32768);pipe.InitBuffer(bcast,R*8*4);pipe.InitBuffer(rb,R*4);
    auto x=xb.Get<float>();auto h=hb.Get<bfloat16_t>();auto mask=maskb.Get<float>();
    auto max=mb.Get<float>(),sum=sb.Get<float>(),ex=eb.Get<float>(),bc=bcast.Get<float>();
    uint32_t core=GetBlockIdx()/2,sub=GetBlockIdx()%2,tasks=Tasks(p);
    uint64_t ws=p.workspace+static_cast<uint64_t>(core)*WORKSPACE;
    for(uint32_t task=core;task<tasks;task+=GetBlockNum()) {
        Selection s(p,task);uint32_t half=(s.rows+1)/2,begin=sub*half,nr=begin<s.rows?s.rows-begin:0;if(nr>half)nr=half;
        Gather(p,s,sub,gb.Get<bfloat16_t>(),ixb.Get<int32_t>(),mask,tmpb.Get<float>(),bitsb.Get<uint8_t>(),Gm<bfloat16_t>(ws+KV));
        uint64_t row=static_cast<uint64_t>(s.token)*8+begin;
        if(nr) {
            DataCopyPad(max,Gm<float>(p.sinks)[begin%8],DataCopyExtParams{1,nr*4,0,0,0},DataCopyPadExtParams<float>{});
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);Duplicate(sum,1.0f,R);
        }
        CrossCoreWaitFlag(SCORE_READY);
        if(nr) {
            DataCopy(x,Gm<float>(ws+SCORE)[begin*N],nr*N);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Muls(x,x,p.softmaxScale,nr*N);PipeBarrier<PIPE_V>();
            for(uint32_t r=0;r<nr;++r)Add(x[r*N],x[r*N],mask,N);PipeBarrier<PIPE_V>();
            SoftMaxShapeInfo shape{nr,N,nr,N};auto tiling=SoftMaxFlashV2TilingFunc(shape,4,4,32768,true,false);
            SoftmaxFlashV2<float,true,true,false,false,SM_CFG>(x,sum[R],max[R],x,ex,sum,max,sm.Get<uint8_t>(),tiling,shape);
            PipeBarrier<PIPE_V>();Cast(h,x,RoundMode::CAST_ROUND,nr*N);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopy(Gm<bfloat16_t>(ws+P)[begin*N],h,nr*N);
        }
        CrossCoreSetFlag<2,PIPE_MTE3>(P_READY);CrossCoreWaitFlag(PV_READY);
        if(nr) {
            DataCopy(x,Gm<float>(ws+PV)[begin*D],nr*D);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Brcb(bc,sum[R],1,{1,8});PipeBarrier<PIPE_V>();
            BinaryRepeatParams rp{};rp.dstBlkStride=rp.src0BlkStride=1;rp.src1BlkStride=0;
            rp.dstRepStride=rp.src0RepStride=D/8;rp.src1RepStride=1;
            for(uint32_t col=0;col<D;col+=64)Div(x[col],x[col],bc,64,nr,rp);
            PipeBarrier<PIPE_V>();Cast(h,x,RoundMode::CAST_RINT,nr*D);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopy(Gm<bfloat16_t>(p.out)[row*D],h,nr*D);
            if(p.lse) {
                auto r=rb.Get<float>();Ln(r,sum[R],nr);PipeBarrier<PIPE_V>();Add(r,r,max[R],nr);
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                DataCopyPad(Gm<float>(p.lse)[row],r,DataCopyExtParams{1,nr*4,0,0,0});
            }
            PipeBarrier<PIPE_ALL>();
        }
    }
}
}
extern "C" __global__ __aicore__ void bf16_mtp_shared_slots(SharedKVArgs p) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if ASCEND_IS_AIC {Cube(p);}if ASCEND_IS_AIV {Vector(p);}
}
extern "C" void launch_sharedkv(uint32_t blocks,void* stream,const SharedKVArgs* args) {
    bf16_mtp_shared_slots<<<blocks,nullptr,stream>>>(*args);
}
