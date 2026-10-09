"""Structured result ingestion, with an explicit historical-data adapter."""
import argparse
import json
from pathlib import Path
from .results import document, emit


def legacy_document(data, path):
    """Preserve historical metadata without inventing missing samples or provenance."""
    return document('historical:'+path.stem,[],
                    dict(representative=False,historical=True,source=str(path),legacy_data=data),
                    environment={'historical':True,'current_measurement':False})


def load(path, include_legacy=False):
    data=json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get('schema')=='orthon-benchmark' and data.get('schema_version')==1:
        return data
    if not include_legacy:
        raise ValueError(f'{path}: historical schema requires --include-legacy')
    return legacy_document(data,Path(path))


def run(name,argv=None):
    p=argparse.ArgumentParser(description=f'{name}: read structured results; no console scraping')
    p.add_argument('files',nargs='*',type=Path)
    p.add_argument('--results-dir',type=Path)
    p.add_argument('--include-legacy',action='store_true')
    p.add_argument('--quick',action='store_true',help='smoke ingestion; representative=false')
    p.add_argument('--json')
    p.add_argument('--markdown',type=Path)
    args=p.parse_args(argv)
    paths=args.files+([] if args.results_dir is None else sorted(args.results_dir.glob('*.json')))
    sources=[load(path,args.include_legacy) for path in paths]
    cases=[dict(c,source_benchmark=s['benchmark'],source_provenance=s['provenance'])
           for s in sources for c in s['cases']]
    result=document(name,cases,dict(representative=False,kind='structured-report',
                    include_legacy=args.include_legacy,sources=[str(p) for p in paths],
                    historical_records=[s['config'] for s in sources if s['config'].get('historical')]),mode='report')
    emit(result,args.json)
    if args.markdown:
        lines=['# Structured benchmark observations','',
               'Each measurement retains its own recorded environment. Historical inputs are explicitly marked.', '',
               '| Case | Implementation | Device | Median | IQR | Units | Correctness |',
               '| --- | --- | --- | --- | --- | --- | --- |']
        for c in cases:
            if 'median' in c:
                lines.append('| '+' | '.join(str(c.get(k,'')) for k in ['name','implementation','device','median','iqr','units','correctness'])+' |')
        args.markdown.parent.mkdir(parents=True,exist_ok=True)
        args.markdown.write_text('\n'.join(lines)+'\n',encoding='utf-8')
