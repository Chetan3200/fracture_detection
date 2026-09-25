"""Shared paths and frozen experiment inputs. Standard library only.

Precedence: CLI overrides (in each entry point) > process environment > .env > defaults.
Relative paths, including paths from .env, are anchored to the project root, not CWD.
The small .env reader accepts KEY=value, quoted values, comments and optional
'export'. It does not execute shell commands or expand $VARIABLE expressions.
Loading configuration does not create directories, import ML libraries or contact HF.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import hashlib
import os
import re
import shlex

PROJECT_ROOT = Path(__file__).resolve().parent
MANIFEST_SHA256 = "1719f37f442512c3c4fcab8350bedc0f27ff2dcebc082761299912b897747034"
CANDIDATE_CSV_SHA256 = "5af4772bf275a4e84a815670d2988fb0408119ea05e0b2ffe0cb3251fdf1e66d"
HARD_NEGATIVE_POOL_SHA256 = "2db84c52f39debc56ad6d6bdae9428ef8d6f2589e7e208781bcaa5f6ddf1a719"
HARD_NEGATIVE_COUNT = 178
HARD_NEGATIVE_WEIGHT = 3.0
DATASET_SOURCE = "jasonroggy/grazpedwri-dx/versions/1"
# Stable names inside sharded backups, independent of each machine's DATA_DIR.
DATA_ARCHIVE_FOLDERS = ("data/grazpedwri_yolo", "data/grazpedwri_medgemma")
SPLIT_COUNTS = {
    "train": {"positive": 9479, "negative": 3961, "excluded": 803},
    "val": {"positive": 2045, "negative": 850, "excluded": 163},
    "test": {"positive": 2026, "negative": 846, "excluded": 154},
}
YOLO_BATCH_SIZES = {640: 35, 960: 14}
YOLO_EPOCHS = 100
YOLO_SEED = 43
BACKUP_EVERY = 10
UPLOAD_RETRIES = 3
EXPECTED_ULTRALYTICS = "8.4.152"
EXPECTED_TORCH = "2.11.0+cu128"

# Frozen MedGemma recipe; machine/GPU choices remain environment/CLI settings.
MEDGEMMA_BASE_REVISION = "91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b"
MEDGEMMA_DEFAULTS = {
    "epochs": 3, "lr": 1e-4, "seed": 42, "lora_rank": 16, "attention": "sdpa",
    "max_seq_len": 2048, "generation_samples": 16, "max_new_tokens": 768,
    "backup_steps": 100,
}
MEDGEMMA_GPU_DEFAULTS = {
    "CUDA_VISIBLE_DEVICES": "0", "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "NCCL_P2P_DISABLE": "1", "NCCL_IB_DISABLE": "1",
}

# Configuration frontends pass these explicitly to the hash-pinned evaluators.
# In test/inference protocol mode, saved prediction settings take precedence.
EVALUATION_DEFAULTS = {
    "imgsz": 640, "batch": 8, "bootstraps": 1000,
    "bootstrap_seed": 2026, "benchmark_runs": 100,
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(value, root=PROJECT_ROOT):
    """Resolve user paths against the declared project root, including '~'."""
    result = Path(value).expanduser()
    return (result if result.is_absolute() else Path(root) / result).resolve()


def read_env_file(filename, required=False):
    """Read simple .env assignments without evaluation or secret-bearing errors."""
    filename = Path(filename)
    if not filename.exists():
        if required:
            raise FileNotFoundError(f"Environment file not found: {filename}")
        return {}
    if not filename.is_file():
        raise ValueError(f"Environment path is not a file: {filename}")
    values = {}
    for number, line in enumerate(filename.read_text(encoding="utf-8-sig").splitlines(), 1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if text.startswith("export "):
            text = text[7:].lstrip()
        key, separator, value = text.partition("=")
        key = key.strip()
        if not separator or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) is None:
            raise ValueError(f"Invalid .env assignment at {filename}:{number}")
        try:
            parts = shlex.split(value, comments=True, posix=True)
        except ValueError:
            raise ValueError(f"Invalid .env quoting at {filename}:{number}") from None
        if len(parts) > 1:
            raise ValueError(f"Quote values containing spaces at {filename}:{number}")
        values[key] = parts[0] if parts else ""
    return values


@dataclass(frozen=True)
class ProjectConfig:
    project_root: Path
    env_file: Path
    data_dir: Path
    runs_dir: Path
    cache_dir: Path
    manifest_path: Path
    candidates_path: Path
    hf_repo_id: Optional[str]
    medgemma_hf_repo_id: Optional[str]
    dataset_hf_repo_id: Optional[str]
    hf_home: Optional[Path]
    kagglehub_cache: Optional[Path]
    _dotenv_values: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def yolo_data_dir(self):
        return self.data_dir / "grazpedwri_yolo"

    @property
    def medgemma_data_dir(self):
        return self.data_dir / "grazpedwri_medgemma"

    def setting(self, name, default=None):
        """Read a requested machine setting without activating or dumping .env."""
        return os.environ.get(name, self._dotenv_values.get(name, default))

    def activate_environment(self):
        """Apply .env only when executing a workflow, never while printing config.

        Existing process variables win. SDK paths are made absolute before SDK
        import. No HF_HOME default is forced, preserving the existing HF login.
        """
        for key, value in self._dotenv_values.items():
            # A blank optional SDK path in .env means "use the existing SDK default",
            # not "store credentials/cache in the current working directory".
            if key in {"HF_HOME", "KAGGLEHUB_CACHE"} and not value.strip():
                continue
            os.environ.setdefault(key, value)
        if self.hf_home is not None:
            os.environ["HF_HOME"] = str(self.hf_home)
        if self.kagglehub_cache is not None:
            os.environ["KAGGLEHUB_CACHE"] = str(self.kagglehub_cache)

    def public_dict(self):
        """Only allowlisted, non-secret settings may enter logs/run metadata."""
        return {
            "project_root": str(self.project_root),
            "env_file": str(self.env_file),
            "data_dir": str(self.data_dir),
            "runs_dir": str(self.runs_dir),
            "cache_dir": str(self.cache_dir),
            "manifest_path": str(self.manifest_path),
            "candidates_path": str(self.candidates_path),
            "hf_repo_id": self.hf_repo_id,
            "medgemma_hf_repo_id": self.medgemma_hf_repo_id,
            "dataset_hf_repo_id": self.dataset_hf_repo_id,
            "hf_home": str(self.hf_home) if self.hf_home else None,
            "kagglehub_cache": str(self.kagglehub_cache) if self.kagglehub_cache else None,
        }


def load_config(project_root=None, env_file=None, environ=None):
    """Read a root-level .env without mutating the process or creating files."""
    root = Path(project_root).expanduser().resolve() if project_root is not None else PROJECT_ROOT
    dotenv_path = resolve_path(env_file, root) if env_file is not None else root / ".env"
    dotenv = read_env_file(dotenv_path, required=env_file is not None)
    values = {**dotenv, **(os.environ if environ is None else environ)}

    def location(key, default):
        value = values.get(key, "").strip() or default
        return resolve_path(value, root)

    def optional_location(key):
        value = values.get(key, "").strip()
        return resolve_path(value, root) if value else None

    def repository(key):
        return values.get(key, "").strip() or None

    return ProjectConfig(
        project_root=root, env_file=dotenv_path,
        data_dir=location("DATA_DIR", "data"),
        runs_dir=location("RUNS_DIR", "runs"),
        cache_dir=location("CACHE_DIR", ".cache"),
        manifest_path=location("MANIFEST_PATH", "split_manifest.csv"),
        candidates_path=location("CANDIDATES_PATH", "hard_negatives_review/ranked_negatives.csv"),
        hf_repo_id=repository("HF_REPO_ID"),
        medgemma_hf_repo_id=repository("MEDGEMMA_HF_REPO_ID"),
        dataset_hf_repo_id=repository("DATASET_HF_REPO_ID"),
        hf_home=optional_location("HF_HOME"),
        kagglehub_cache=optional_location("KAGGLEHUB_CACHE"),
        _dotenv_values=dotenv,
    )


def dataset_directories(project):
    """Map the portable archive layout to configured local dataset directories."""
    return dict(zip(DATA_ARCHIVE_FOLDERS, (project.yolo_data_dir, project.medgemma_data_dir)))
