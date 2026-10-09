"""The sole UTF-8 strict JSON result writer."""
import json
from pathlib import Path
from .environment import provenance


def case_result(name, stats, *, shape, implementation, device, dtype='float32',
                mode='forward', warmup=25, correctness='not-compared', notes='', **extra):
    return dict(name=name, shape=shape, implementation=implementation, device=device,
                dtype=dtype, mode=mode, warmup=warmup, correctness=correctness,
                notes=notes, **stats.as_dict(), **extra)


def document(benchmark, cases, config, *, mode='benchmark', environment=None):
    return dict(schema='orthon-benchmark', schema_version=1, benchmark=benchmark,
                mode=mode, provenance=provenance() if environment is None else environment,
                config=config, cases=cases)


def write_json(path, payload):
    encoded = json.dumps(payload, indent=2, allow_nan=False, ensure_ascii=False)+'\n'
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(encoded, encoding='utf-8')


def emit(payload, path=None):
    print(f"{payload['benchmark']} mode={payload['mode']} representative={str(payload['config'].get('representative',True)).lower()}")
    for c in payload['cases']:
        if 'median' in c:
            print(f"  {c['name']} {c['implementation']} {c['device']} {c['dtype']}: "
                  f"median={c['median']:.6f} IQR={c['iqr']:.6f} {c['units']} "
                  f"correctness={c['correctness']} verdict={c.get('verdict','observation')}")
        else:
            print(f"  {c}")
    if path:
        write_json(path,payload)
