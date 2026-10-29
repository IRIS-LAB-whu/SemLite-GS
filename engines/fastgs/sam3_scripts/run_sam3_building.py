#!/usr/bin/env python3
"""Extract a building mask from a DOM with tiled SAM3 inference."""

from __future__ import annotations

import argparse
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

PROMPTS = ("building", "buildings", "roof")
TILE_SIZE = 2000
STRIDE = 1000
MIN_TILE_SIZE = 128
SCORE_THRESHOLD = 0.10
DEFAULT_MAX_HOLE_AREA = 500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--mask-name", default="TOM_mask_sam3.png")
    parser.add_argument("--raw-mask-name", default="TOM_mask_sam3_raw.png")
    parser.add_argument("--overlay-name", default="building_overlay.png")
    parser.add_argument(
        "--overlay-color",
        choices=("red", "green", "blue"),
        default="red",
        help="Colour used for the review overlay; label PNG values are unchanged.",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=list(PROMPTS),
        help="Open-vocabulary prompts to union. Defaults: building buildings roof.",
    )
    parser.add_argument(
        "--max-hole-area",
        type=int,
        default=DEFAULT_MAX_HOLE_AREA,
        help="填充面积不超过该值的建筑内部孔洞；0 表示填充所有孔洞",
    )
    parser.add_argument(
        "--postprocess-only",
        action="store_true",
        help="跳过 SAM3 推理，仅对已有 --mask-name 执行填孔并更新 overlay",
    )
    parser.add_argument("--sam3-root", type=Path, default=Path(__file__).resolve().parents[2] / "sam3")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--bpe-path", type=Path)
    args = parser.parse_args()
    if not args.postprocess_only:
        from semlite_gs.checkpoints import resolve_sam3
        args.checkpoint = resolve_sam3(args.checkpoint)
    if args.bpe_path is None:
        args.bpe_path = (
            args.sam3_root / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
        )
    return args


def load_processor(sam3_root: Path, checkpoint: Path, bpe_path: Path, device: str):
    sys.path.insert(0, str(sam3_root.resolve()))
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model(
        checkpoint_path=str(checkpoint.resolve()),
        bpe_path=str(bpe_path.resolve()),
        device=device,
    )
    return Sam3Processor(model)


def segment_tiles(
    image: np.ndarray,
    processor,
    prompts: tuple[str, ...],
) -> tuple[np.ndarray, int, int]:
    height, width = image.shape[:2]
    output = np.zeros((height, width), dtype=np.uint8)
    tile_count = 0
    detection_count = 0

    for y0 in range(0, height, STRIDE):
        for x0 in range(0, width, STRIDE):
            y1 = min(y0 + TILE_SIZE, height)
            x1 = min(x0 + TILE_SIZE, width)
            tile = image[y0:y1, x0:x1]
            if min(tile.shape[:2]) < MIN_TILE_SIZE:
                continue

            tile_union = np.zeros(tile.shape[:2], dtype=bool)
            tile_detections = 0

            amp_context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if torch.cuda.is_available()
                else nullcontext()
            )
            with amp_context:
                state = processor.set_image(Image.fromarray(tile))
                for prompt in prompts:
                    prediction = processor.set_text_prompt(state=state, prompt=prompt)
                    masks = prediction["masks"]
                    scores = prediction["scores"]
                    masks = masks[scores >= SCORE_THRESHOLD]
                    if masks.shape[0] == 0:
                        continue

                    tile_detections += int(masks.shape[0])
                    tile_union |= masks.squeeze(1).any(dim=0).detach().cpu().numpy()

            output[y0:y1, x0:x1] = np.maximum(
                output[y0:y1, x0:x1],
                tile_union.astype(np.uint8) * 255,
            )
            tile_count += 1
            detection_count += tile_detections
            print(
                f"tile {tile_count}: ({x0},{y0})-({x1},{y1}) "
                f"detections={tile_detections}"
            )

    return output, tile_count, detection_count


def save_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    path: Path,
    color_name: str = "red",
) -> None:
    overlay = image.copy()
    foreground = mask > 0
    color = np.zeros_like(overlay)
    channel = {"red": 0, "green": 1, "blue": 2}[color_name]
    color[..., channel] = 255
    overlay[foreground] = (overlay[foreground] * 0.55 + color[foreground] * 0.45).astype(
        np.uint8
    )
    Image.fromarray(overlay).save(path)


def fill_mask_holes(
    mask: np.ndarray,
    max_hole_area: int,
) -> tuple[np.ndarray, int, int]:
    binary = mask > 0
    all_filled = ndimage.binary_fill_holes(binary)
    holes = all_filled & (~binary)
    labels, count = ndimage.label(
        holes,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    if count == 0:
        return mask.astype(np.uint8), 0, 0

    areas = np.bincount(labels.ravel())
    accepted = np.ones(count + 1, dtype=bool)
    accepted[0] = False
    if max_hole_area > 0:
        accepted &= areas <= int(max_hole_area)
    fill_pixels = accepted[labels]
    result = binary | fill_pixels
    return (
        result.astype(np.uint8) * 255,
        int(accepted[1:].sum()),
        int(fill_pixels.sum()),
    )


def main() -> None:
    args = parse_args()
    image_path = args.image.resolve()
    output_dir = (args.out_dir or image_path.parent).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    Image.MAX_IMAGE_PIXELS = None
    image = np.asarray(Image.open(image_path).convert("RGB"))
    mask_path = output_dir / args.mask_name
    raw_mask_path = output_dir / args.raw_mask_name
    overlay_path = output_dir / args.overlay_name

    if args.postprocess_only:
        if not mask_path.exists():
            raise FileNotFoundError(f"找不到待处理 mask：{mask_path}")
        source_mask_path = raw_mask_path if raw_mask_path.exists() else mask_path
        raw_mask = np.asarray(Image.open(source_mask_path).convert("L"))
        tile_count = 0
        detection_count = 0
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading SAM3 on {device}")
        processor = load_processor(
            args.sam3_root,
            args.checkpoint,
            args.bpe_path,
            device,
        )
        raw_mask, tile_count, detection_count = segment_tiles(
            image,
            processor,
            tuple(args.prompts),
        )

    Image.fromarray(raw_mask).save(raw_mask_path)
    mask, filled_holes, filled_pixels = fill_mask_holes(
        raw_mask,
        args.max_hole_area,
    )
    Image.fromarray(mask).save(mask_path)
    save_overlay(image, mask, overlay_path, args.overlay_color)

    print(f"Saved raw mask: {raw_mask_path}")
    print(f"Saved mask: {mask_path}")
    print(f"Saved overlay: {overlay_path}")
    print(
        f"Filled holes: {filled_holes}; filled pixels: {filled_pixels}; "
        f"max hole area: {args.max_hole_area}"
    )
    print(f"Processed tiles: {tile_count}; kept detections: {detection_count}")


if __name__ == "__main__":
    main()
