import os
from pathlib import Path


def resolve_sam3(explicit=None, cache_dir=None):
    candidate = explicit or os.environ.get('SEMLITE_SAM3_CHECKPOINT')
    if candidate:
        path = Path(candidate).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError('Configured SAM3 checkpoint is absent: ' + str(path))
        return path
    from huggingface_hub import hf_hub_download
    options = dict(repo_id='facebook/sam3', filename='sam3.pt')
    if cache_dir:
        options['cache_dir'] = str(Path(cache_dir).expanduser())
    try:
        return Path(hf_hub_download(**options, local_files_only=True))
    except Exception:
        try:

            return Path(hf_hub_download(**options))
        except Exception as exc:
            raise RuntimeError('SAM3 was not found in the cache and automatic download failed. '
                               'For the gated facebook/sam3 repository, accept the model terms '
                               'and authenticate with HF_TOKEN or `hf auth login`; then rerun. '
                               'You can also set SEMLITE_SAM3_CHECKPOINT to an existing sam3.pt. '
                               'Weights are never copied into this project.') from exc
