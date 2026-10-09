"""Canonical allocator reuse and scratch accounting benchmark."""
import argparse
from dataclasses import asdict
import orthon as o
import orthon._diagnostics as diag
from benchmarks._support.cli import add_options, timing_config
from benchmarks._support.environment import cuda_available, synchronizer
from benchmarks._support.timing import measure
from benchmarks._support.results import case_result, document, emit


def main(argv=None):
    p=add_options(argparse.ArgumentParser(description=__doc__))
    p.add_argument('--device',choices=['cpu','cuda:0'],default='cpu')
    p.add_argument('--shape',nargs='+',type=int,default=[256,256])
    args=p.parse_args(argv)
    if args.device.startswith('cuda') and not cuda_available(args.device):
        p.error('no usable CUDA device')
    cfg=timing_config(args)
    shape=[16,16] if args.quick else args.shape
    cases=[]
    enabled,limit=diag.memory_pool_enabled(),diag.memory_pool_max_cached_bytes()
    try:
        for use_pool in [False,True]:
            diag.memory_pool_clear_cache()
            diag.memory_pool_reset_stats()
            diag.memory_pool_set_enabled(use_pool)
            stats=measure(lambda:o.zeros(shape).to(args.device),synchronizer(args.device),cfg)
            counts=diag.memory_pool_stats()
            scratch=diag.memory_pool_scratch_stats(args.device)
            cases.append(case_result('pool-'+str(use_pool),stats,shape=shape,
                implementation='orthon',device=args.device,warmup=cfg.warmup,
                notes='counters include warmup/calibration; includes zero fill and device transfer',
                pool=counts,scratch=scratch,cache_capacity=limit))
            inputs=o.ones(shape).to(args.device)
            diag.memory_pool_reset_stats()
            reductions=measure(lambda:(o.sum(inputs),o.mean(inputs),o.max(inputs)),
                               synchronizer(args.device),cfg)
            cases.append(case_result('reduction-scratch-'+str(use_pool),reductions,shape=shape,
                implementation='orthon',device=args.device,warmup=cfg.warmup,
                notes='one call contains sum, mean and max; actual runtime scratch leases',
                pool=diag.memory_pool_stats(),scratch=diag.memory_pool_scratch_stats(args.device)))
        # Retained manual cursor is diagnostic-only; runtime leases are unaffected.
        diag.memory_pool_reset_scratch(args.device)
        diag.memory_pool_scratch_allocate(256,args.device)
        scratch=diag.memory_pool_scratch_stats(args.device)
        for c in cases:
            c['scratch_probe']=scratch
    finally:
        diag.memory_pool_reset_scratch(args.device)
        diag.memory_pool_set_enabled(enabled)
        diag.memory_pool_set_max_cached_bytes(limit)
        diag.memory_pool_clear_cache()
    emit(document('memory_pool',cases,dict(representative=not args.quick,timing=asdict(cfg))),args.json)


if __name__=='__main__':
    main()
