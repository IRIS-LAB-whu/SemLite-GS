import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
import torch


class TrainingTimer:
    def __init__(self, model_path, enabled=False, barrier='', worker='0'):
        self.enabled = enabled
        self.root = Path(model_path)
        self.barrier = Path(barrier) if barrier else None
        self.worker = worker
        self.entered_epoch = time.time()
        self.save_seconds = 0.

    def track_gpu(self):
        if self.enabled:
            torch.cuda.reset_peak_memory_stats()

    def ready(self):
        if not self.enabled:
            return
        torch.cuda.synchronize()
        self.ready_epoch = time.time()
        if self.barrier:
            self.barrier.mkdir(parents=True, exist_ok=True)
            target = self.barrier / (self.worker + '.ready.json')
            temp = target.with_suffix('.tmp')
            temp.write_text(json.dumps({'pid':os.getpid(),'ready_epoch':self.ready_epoch}))
            temp.replace(target)
            while not (self.barrier / 'release').exists():
                if (self.barrier / 'abort').exists():
                    raise RuntimeError('Benchmark wave aborted before training')
                if time.time()-self.ready_epoch > 1800:
                    raise TimeoutError('Benchmark barrier did not release in 30 minutes')
                time.sleep(.05)
        self.started_epoch = time.time()
        self.started = time.perf_counter()
        (self.root/'timing_start.json').write_text(json.dumps({
            'ready_epoch':self.ready_epoch,'train_start_epoch':self.started_epoch,
            'loading_internal_seconds':self.ready_epoch-self.entered_epoch}))

    @contextmanager
    def saving(self):
        if not self.enabled:
            yield
            return
        torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            torch.cuda.synchronize()
            self.save_seconds += time.perf_counter()-start

    def finish(self, iterations):
        if not self.enabled:
            return
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-self.started
        data = dict(iterations=iterations,ready_epoch=self.ready_epoch,
            train_start_epoch=self.started_epoch,train_end_epoch=time.time(),
            loading_internal_seconds=self.ready_epoch-self.entered_epoch,
            barrier_wait_seconds=self.started_epoch-self.ready_epoch,
            training_seconds_including_save=elapsed,
            training_seconds_excluding_save=elapsed-self.save_seconds,
            save_seconds=self.save_seconds,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved())
        (self.root/'training_timing.json').write_text(json.dumps(data,indent=2))
