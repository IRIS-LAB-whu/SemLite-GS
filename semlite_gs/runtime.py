from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ENGINES = {'fastgs': ROOT/'engines/fastgs', 'pgsr': ROOT/'engines/semantic_pgsr', 'sam3': ROOT/'engines/sam3', 'mesh': ROOT}


@lru_cache(maxsize=None)
def code_fingerprint(engine):
    roots=[ROOT/'semlite_gs']
    if engine!='mesh':roots.append(ENGINES[engine])
    h=hashlib.sha256()
    for root in roots:
        for p in sorted(root.rglob('*')):
            if p.suffix not in ('.py','.cpp','.cu','.h','.hpp','.cuh') or 'build' in p.parts: continue
            h.update(str(p.relative_to(ROOT)).encode());h.update(p.read_bytes())
    return h.hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def engine_environment(engine, gpu=None, cpu_threads=1):
    env = os.environ.copy()
    paths = [ROOT, ENGINES[engine]]
    if engine == 'pgsr':
        paths += [ENGINES[engine]/'submodules'/s for s in
                  ('diff-plane-rasterization','simple-knn','pgsr-pipeline-ops','fused-ssim','pgsr-view-batch')]
    if engine == 'fastgs':
        paths += [ENGINES[engine]/'submodules'/s for s in ('diff-gaussian-rasterization_fastgs','diff-gaussian-rasterization_fastgs_ortho','diff-gaussian-rasterization-semantic','simple-knn','fused-ssim')]
    env.update(PYTHONPATH=os.pathsep.join(map(str, paths)), PYTHONNOUSERSITE='1',
               OMP_NUM_THREADS=str(cpu_threads), OPENBLAS_NUM_THREADS=str(cpu_threads),
               MKL_NUM_THREADS=str(cpu_threads), PYTHONUNBUFFERED='1')
    if gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    return env


def run_command(config, name, engine, arguments, outputs=(), gpu=None):

    out = Path(config['output']); state = out/'logs'/f'{name}.json'
    python = config.get('python', {}).get(engine, sys.executable)
    command = [python] + list(map(str, arguments))
    signature = hashlib.sha256(json.dumps([command,config,code_fingerprint(engine)],sort_keys=True).encode()).hexdigest()
    if config.get('resume', True) and state.exists():
        old = json.loads(state.read_text())
        if old.get('signature') == signature and old.get('state') == 'complete' and all(Path(p).exists() for p in outputs):
            print('Reuse', name, flush=True); return old
    state.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    record = dict(stage=name, state='running', command=command, cwd=str(ENGINES[engine]),
                  signature=signature, started=started, outputs=list(map(str, outputs)))
    write_json(state,record)
    print('Run',name,flush=True)
    try:
        with state.with_suffix('.log').open('w') as stream:
            subprocess.run(command, cwd=ENGINES[engine], env=engine_environment(engine,gpu,config.get('cpu_threads',1)),
                           stdout=stream, stderr=subprocess.STDOUT, check=True)
        if not all(Path(p).exists() for p in outputs):
            raise RuntimeError('Missing declared stage output for '+name)
        record['state']='complete'
    except BaseException as exc:
        record.update(state='failed',error=str(exc)); raise
    finally:
        record['seconds']=time.time()-started;write_json(state,record)
    return record
