from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Optional, Union


PARTITION_MODES = ("none", "level0", "level1", "level2")


def add_partition_mode_argument(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--partition_mode",
        choices=PARTITION_MODES,
        default="none",
        help="none keeps the original instance mask; levels select hierarchical_masks/<level>.",
    )


def resolve_mask_source(
    partition_mode: str,
    instance_mask: Union[str, Path],
    hierarchical_masks: Optional[Union[str, Path]] = None,
) -> Path:

    if partition_mode not in PARTITION_MODES:
        raise ValueError(f"Unknown partition_mode {partition_mode!r}; choose from {PARTITION_MODES}")
    original = Path(instance_mask)
    if partition_mode == "none":
        return original
    if hierarchical_masks is None:
        raise ValueError("hierarchical_masks is required unless partition_mode=none")
    selected = Path(hierarchical_masks) / partition_mode
    if original.suffix:
        selected = selected / original.name
    return selected
