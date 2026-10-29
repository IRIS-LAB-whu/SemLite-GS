import random
import numpy as np
import torch


def backward_with_cache_retry(compute, fallback, clear_gradients, gpu_cache):
    if not gpu_cache:
        return compute()
    rng = (random.getstate(),np.random.get_state(),torch.get_rng_state(),
           torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    try:
        return compute()
    except torch.cuda.OutOfMemoryError:
        pass

    clear_gradients()
    fallback()
    random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2])
    if rng[3] is not None:
        torch.cuda.set_rng_state_all(rng[3])
    return compute()
