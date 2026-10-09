"""Deterministic microbenchmark setup. Allocation/conversion precedes timing."""
import argparse
from dataclasses import asdict
import numpy as np
import orthon as o
import orthon._diagnostics as diag
from .cli import add_options, timing_config
from .environment import cuda_available, synchronizer, global_warmup, temporary_environment
from .timing import measure, measure_pair, comparison
from .results import case_result, document, emit


def array(t):
    return np.asarray(t.to('cpu').tolist()).reshape(tuple(t.shape))


def run(kind, argv=None):
    p = add_options(argparse.ArgumentParser(description=f'{kind}: deterministic operator measurement'))
    p.add_argument('--device',choices=['cpu','cuda:0'],default='cpu')
    p.add_argument('--shape',nargs='+',type=int)
    p.add_argument('--dtype',choices=['float32','float64','float16','bfloat16'],default='float32')
    p.add_argument('--configs',default='',help='Conv2d presets, comma-separated: cifar_conv1,cifar_conv2,synthetic,minivgg_32,minivgg_64,minivgg_128')
    p.add_argument('--batch',type=int,default=1,help='matmul batch count; 1 selects rank-2')
    p.add_argument('--torch',action='store_true',help='paired eager PyTorch reference')
    args=p.parse_args(argv)
    if args.device.startswith('cuda') and not cuda_available(args.device):
        p.error('no usable Orthon CUDA device')
    cfg=timing_config(args)
    global_warmup(args.device,args.quick)
    rng=np.random.default_rng(1919)
    size=(args.shape or ([32] if args.quick else [256]))[0]
    cases=[]
    def tensor(a,grad=False):
        host=np.ascontiguousarray(a,dtype=np.float16 if args.dtype=='float16' else np.float64 if args.dtype=='float64' else np.float32)
        t=(o.from_numpy(host,dtype='bfloat16') if args.dtype=='bfloat16' else o.from_numpy(host)).to(args.device)
        t.requires_grad_(grad)
        return t
    def record(name, fn, shape, expected=None, reference=None, notes='', mode='forward', check=None):
        correct=True if check is None else check()
        if expected is not None:
            correct=bool(np.allclose(array(fn()),expected,rtol=.03 if args.dtype!='float32' else 3e-4,atol=.03 if args.dtype!='float32' else 3e-4))
        sync=synchronizer(args.device)
        if reference is not None:
            a,b=measure_pair(fn,reference,sync,synchronizer(args.device,'torch'),cfg)
            decision=comparison(a,b,correct)
            for impl,stats in [('orthon',a),('pytorch-eager',b)]:
                cases.append(case_result(name,stats,shape=shape,implementation=impl,device=args.device,dtype=args.dtype,
                                        mode=mode,warmup=cfg.warmup,correctness='pass' if correct else 'fail',notes=notes,**decision))
        else:
            stats=measure(fn,sync,cfg)
            cases.append(case_result(name,stats,shape=shape,implementation='orthon',device=args.device,dtype=args.dtype,
                                     mode=mode,warmup=cfg.warmup,correctness=('pass' if correct else 'fail') if expected is not None or check is not None else 'not-compared',notes=notes))
    torch=None
    if args.torch:
        import torch
    if kind in ('matmul','fusion','no_grad'):
        matrix_shape=(args.batch,size,size) if kind=='matmul' and args.batch>1 else (size,size)
        a32=rng.normal(0,.05,matrix_shape).astype('float32')
        b32=rng.normal(0,.05,matrix_shape).astype('float32')
        a,b=tensor(a32,kind=='no_grad'),tensor(b32,kind=='no_grad')
        expected=array(a)@array(b)
        ref=None
        if torch is not None:
            at=torch.tensor(array(a),device=args.device)
            bt=torch.tensor(array(b),device=args.device)
            if args.dtype!='float32':
                at,bt=at.to(getattr(torch,args.dtype)),bt.to(getattr(torch,args.dtype))
            ref=lambda: at@bt
        record('public-matmul',lambda:o.matmul(a,b),list(matrix_shape)+[size],expected,ref)
        if kind=='matmul' and args.device.startswith('cuda'):
            forced=[('custom',diag.batched_matmul_custom if args.batch>1 else diag.matmul_custom)]
            if args.dtype=='float32':
                if args.batch==1:
                    forced.append(('naive',diag.matmul_naive))
            else:forced.append(('cublas',diag.batched_matmul_cublas if args.batch>1 else diag.matmul_cublas))
            for label,fn in forced:
                record(label,lambda fn=fn:fn(a,b),list(matrix_shape)+[size],expected,
                       notes='diagnostic forced path; '+diag.matmul_guard_report(a,b))
        if kind=='fusion':
            bias=tensor(np.zeros(size,dtype='float32'))
            record('separate',lambda:o.relu(o.matmul(a,b)+bias),[size,size,size],np.maximum(expected,0))
            record('fused',lambda:o.fused_linear_relu(a,b,bias),[size,size,size],np.maximum(expected,0))
        if kind=='no_grad':
            with o.no_grad():
                record('no-grad',lambda:o.matmul(a,b),list(matrix_shape)+[size],expected,notes='graph attachment disabled')
        if kind=='no_grad':
            o.manual_seed(1919)
            models=[('mlp',o.nn.Sequential(o.nn.Linear(size,size),o.nn.ReLU(),o.nn.Linear(size,8)),a),
                    ('cnn',o.nn.Sequential(o.nn.Conv2d(3,4,3,padding=1),o.nn.ReLU()),
                     tensor(rng.normal(0,.05,(2,3,8,8)),True))]
            for name,model,inputs in models:
                for param in model.parameters():
                    param.fill_(.01)
                model.to(args.device)
                expected_model=array(model(inputs))
                record(name+'-grad-enabled',lambda:model(inputs),list(inputs.shape),expected_model)
                with o.no_grad():
                    record(name+'-no-grad',lambda:model(inputs),list(inputs.shape),expected_model,
                           notes='model inference with graph attachment disabled')
    elif kind=='conv2d':
        presets=dict(cifar_conv1=(8,3,32,32,8,3),cifar_conv2=(8,8,32,32,8,3),
                     synthetic=(16,16,32,32,16,3),minivgg_32=(128,32,32,32,32,3),
                     minivgg_64=(128,64,16,16,64,3),minivgg_128=(128,128,8,8,128,3))
        selected=args.configs.split(',') if args.configs else ['cifar_conv1']
        if args.shape:
            if len(args.shape)!=6:
                p.error('Conv2d --shape expects N Cin H W Cout K')
            presets['custom']=tuple(args.shape)
            selected=['custom']
        if args.quick:
            presets={name:(2,3,8,8,4,3) for name in selected}
        for config_name in selected:
            if config_name not in presets:
                p.error('unknown Conv2d preset '+config_name)
            n,cin,h,wid,cout,k=presets[config_name]
            x=tensor(rng.normal(0,.05,(n,cin,h,wid)),True)
            w=tensor(rng.normal(0,.05,(cout,cin,k,k)),True)
            b=tensor(np.zeros(cout),True)
            shape=[list(x.shape),list(w.shape)]
            default=lambda:o.conv2d(x,w,b,1,1,k//2,k//2)
            expected=array(default())
            ref=None
            if torch is not None:
                xt=torch.tensor(array(x),device=args.device)
                wt=torch.tensor(array(w),device=args.device)
                bt=torch.tensor(array(b),device=args.device)
                ref=lambda:torch.nn.functional.conv2d(xt,wt,bt,padding=k//2)
                expected=ref().detach().cpu().numpy()
            record(config_name+':default-forward',default,shape,expected,ref)
            variants=[('cpu-direct',diag.conv2d_direct),('cpu-im2col',diag.conv2d_im2col)] if args.device=='cpu' else [
                ('cuda-naive',diag.conv2d_cuda_naive),('cuda-optimized',diag.conv2d_cuda_optimized)]
            for name,fn in variants:
                record(config_name+":"+name,lambda fn=fn:fn(x,w,b,1,1,k//2,k//2),shape,expected,notes='diagnostic forced path')
            def backward():
                for t in [x,w,b]:
                    t.zero_grad()
                o.sum(default()).backward()
            backward()
            expected_grads=[array(t.grad).copy() for t in [x,w,b]]
            def check_backward():
                backward()
                return all(np.allclose(array(t.grad),expected,atol=3e-4,rtol=3e-4)
                           for t,expected in zip([x,w,b],expected_grads))
            record(config_name+':default-backward',backward,shape,notes='forward + all gradients; public autograd',mode='forward+backward',check=check_backward)
            # End-to-end forcing always restores selectors, including exceptions.
            if args.device.startswith('cuda'):
                for value in ['naive','optimized','optimized_v2']:
                    with temporary_environment({'ORTHON_CUDA_CONV2D_GRAD_INPUT':value}):
                        record(config_name+':forced-backward-'+value,backward,shape,notes='diagnostic forced path',mode='forward+backward',check=check_backward)
    elif kind in ('softmax_cross_entropy',):
        a32=rng.normal(0,.05,(size,16)).astype('float32')
        a=tensor(a32)
        exp=np.exp(array(a))
        expected=exp/exp.sum(axis=1,keepdims=True)
        record('softmax',lambda:o.softmax(a),[size,16],expected)
        targets=[i%16 for i in range(size)]
        loss=-np.log(expected[np.arange(size),targets]).mean()
        record('cross-entropy',lambda:o.cross_entropy(a,targets),[size,16],loss)
    else:
        a32=rng.normal(0,.05,(size*size,)).astype('float32')
        a=tensor(a32)
        b=tensor(a32)
        if kind in ('elementwise','ops','cpu_simd'):
            record('add',lambda:o.add(a,b),list(a.shape),array(a)+array(b))
            record('relu',lambda:o.relu(a),list(a.shape),np.maximum(array(a),0))
        if kind=='cpu_simd':
            record('neg',lambda:o.neg(a),list(a.shape),-array(a))
            record('leaky-relu',lambda:o.leaky_relu(a,.1),list(a.shape),np.where(array(a)>0,array(a),array(a)*.1))
            record('mul',lambda:o.mul(a,b),list(a.shape),array(a)*array(b))
            inplace=tensor(a32)
            record('inplace-mul',lambda:inplace.mul_(1.000001),list(a.shape),notes='one continuously mutated buffer')
            record('inplace-add',lambda:inplace.add_(b,alpha=.000001),list(a.shape),notes='one continuously mutated buffer')
        if kind in ('reductions','ops','cpu_simd'):
            record('sum',lambda:o.sum(a),list(a.shape),array(a).sum())
            record('mean',lambda:o.mean(a),list(a.shape),array(a).mean())
        if kind=='dropout':
            o.manual_seed(1919)
            layer=o.nn.Dropout(.2, seed=1919)
            record('training-dropout',lambda:layer(a),list(a.shape),notes='stochastic training mask; seed=1919')
            layer.eval()
            record('evaluation-dropout',lambda:layer(a),list(a.shape),array(a))
    payload=document(kind,cases,dict(representative=not args.quick,timing=asdict(cfg),ordinal=0 if args.device.startswith('cuda') else None,seed=1919))
    emit(payload,args.json)
    return 1 if any(c['correctness']=='fail' for c in cases) else 0
