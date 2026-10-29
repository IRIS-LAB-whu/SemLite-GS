from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager


def ordered_prefetch(function, items, workers=0):
    if workers < 0:
        raise ValueError('IO workers must be nonnegative')
    if not workers:
        for item in items:
            yield function(item)
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = deque()
        for item in items:
            pending.append(pool.submit(function, item))
            if len(pending) >= 2 * workers:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


@contextmanager
def bounded_writer(workers=0):
    if workers < 0:
        raise ValueError('IO workers must be nonnegative')
    if not workers:
        yield lambda fn, *args: fn(*args)
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = deque()
        def submit(fn, *args):
            pending.append(pool.submit(fn, *args))
            if len(pending) >= 2 * workers:
                pending.popleft().result()
        yield submit
        while pending:
            pending.popleft().result()


def camera_batches(items, batch_size, camera_of=lambda item: item):

    if batch_size < 1:
        raise ValueError('Batch size must be positive')
    batch, resolution = [], None
    for item in items:
        camera = camera_of(item)
        size = (camera.image_height, camera.image_width)
        if batch and (size != resolution or len(batch) == batch_size):
            yield batch
            batch = []
        batch.append(item)
        resolution = size
    if batch:
        yield batch
