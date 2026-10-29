from __future__ import annotations

import numpy as np


def instance_colors(instance_ids: np.ndarray) -> np.ndarray:
    values = np.asarray(instance_ids, dtype=np.int64)
    colors = np.empty(values.shape + (3,), dtype=np.uint8)
    colors[..., 0] = (values * 37 + 53) % 255
    colors[..., 1] = (values * 67 + 91) % 255
    colors[..., 2] = (values * 97 + 149) % 255
    colors[colors.sum(axis=-1) == 0] = np.array(
        [255, 255, 255],
        dtype=np.uint8,
    )
    return colors
