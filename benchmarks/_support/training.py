"""Continuous training-region reporting, retaining workload-specific models."""
from contextvars import ContextVar
from contextlib import contextmanager
from .cli import add_options
from .timing import region, Statistics
from .results import case_result, document, emit

_active = ContextVar('benchmark_session',default=None)


def options(parser):
    return add_options(parser,timing=False)


def configure(args):
    if args.quick:
        for key,value in dict(steps=2,resume_steps=1,warmup=1,warmup_steps=1,batch_size=4,
                              input_dim=8,hidden=8,hidden_size=8,output_dim=4,layers=1,
                              channels=2,iterations=2,iters=2,warmup_iters=1,timed_iters=2,depth=1,epochs=1,max_train_batches=1,
                              max_test_batches=1).items():
            if hasattr(args,key):
                setattr(args,key,value)
    session=_active.get()
    if session is not None:
        session['args']=args
    return args


@contextmanager
def measured(name, sync, *, calls=1, device='cpu', implementation='orthon', dtype='float32', config=None):
    with region(sync) as result:
        yield result
    session=_active.get()
    if session is not None:
        config=dict(config or vars(session['args']))
        stats=Statistics((result['seconds']*1000/calls,),calls)
        case=case_result(name,stats,shape=config,implementation=implementation,
            device=device,dtype=dtype,mode='service' if 'save_start' in name or 'load_start' in name else 'training-region',
            warmup=config.get('warmup',config.get('warmup_steps',config.get('warmup_iters',0))), correctness='not-compared',
            notes=result.get('notes','one continuous region of a changing model; no independent samples or statistical winner; checkpoint saves may use filesystem cache'),
            samples_per_sec=result.get('samples_processed',config.get('batch_size',1)*calls)/result['seconds'],
            **result.get('metrics',{}))
        case['units']=result.get('units','ms/call')
        session['cases'].append(case)


def mark_correctness(status, **metrics):
    session=_active.get()
    if session is not None:
        for case in session['cases']:
            case['correctness']=status
            case.update(metrics)



def update_metadata(**values):
    session=_active.get()
    if session is not None:
        session.setdefault('metadata',{}).update(values)
        for case in session['cases']:
            case['shape'].update(values)


def run(main, name):
    session={'cases':[],'args':None}
    token=_active.set(session)
    try:
        main()
        if session['args'] is not None:
            args=session['args']
            config=dict(vars(args),representative=not args.quick,
                sample_semantics='each case is one continuous region; model is not reset between its steps',
                **session.get('metadata',{}))
            for case in session['cases']:
                if case['mode']=='training-region' and case['units']=='ms/call':
                    case['units']='ms/step'
            emit(document(name,session['cases'],config),args.json)
    finally:
        _active.reset(token)
