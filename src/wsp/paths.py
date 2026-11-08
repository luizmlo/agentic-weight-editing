"""Caminhos fixos do experimento (tudo no disco de dados)."""

from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
REFS = Path("/mnt/dados/dev/refs/lm_understand-logs/exp8_math")

BASE_REPO = "Qwen/Qwen3.5-0.8B"
BASE_REV = "2fc06364715b967f1860aea9cf38778875588b17"  # revisão usada no paper
EDITED_REPO = "RikiyaT/Qwen3.5-0.8B-weight-space-programming"


def snapshot(repo: str, revision: str | None = None) -> Path:
    """Caminho local do snapshot (sem rede se já estiver no cache)."""
    try:
        return Path(snapshot_download(repo, revision=revision, local_files_only=True))
    except Exception:
        return Path(snapshot_download(repo, revision=revision))


def safetensors_file(repo: str, revision: str | None = None) -> Path:
    files = sorted(snapshot(repo, revision).glob("*.safetensors"))
    assert len(files) == 1, files
    return files[0]
