"""Utility functions for the RAG pipeline."""
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:
    from flashrag.utils.utils import GPUMisconfigurationError, ensure_cuda_device, get_device
except ModuleNotFoundError as exc:  # pragma: no cover - runtime environment specific
    project_root = Path(__file__).resolve().parent.parent
    flashrag_root = project_root / "FlashRAG"
    if str(flashrag_root) not in sys.path:
        sys.path.insert(0, str(flashrag_root))
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    try:
        from flashrag.utils.utils import GPUMisconfigurationError, ensure_cuda_device, get_device
    except ModuleNotFoundError as inner_exc:  # pragma: no cover - runtime environment specific
        if inner_exc.name == "torch":
            class GPUMisconfigurationError(RuntimeError):
                """Fallback error when torch is unavailable."""

            def ensure_cuda_device(device_index: int) -> None:
                raise GPUMisconfigurationError(
                    "PyTorch is required for CUDA operations but is not installed in the current environment."
                )

            def get_device() -> str:
                raise GPUMisconfigurationError(
                    "PyTorch is required for CUDA operations but is not installed in the current environment."
                )
        else:
            raise

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True  # force reconfiguration
)
logger = logging.getLogger(__name__)

# ensure immediate flushing
for handler in logger.handlers:
    handler.flush()
sys.stdout.flush()


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL file and return list of dictionaries."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line in {file_path}: {e}")
    return data


def save_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
    """Save list of dictionaries to JSONL file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def get_timestamp() -> str:
    """Get current timestamp in format YYYYMMDD_HHMMSS."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def validate_corpus_format(doc: Dict[str, Any]) -> bool:
    """Validate that document has required fields."""
    return "id" in doc and "contents" in doc and isinstance(doc["contents"], str)


def validate_chunk_format(chunk: Dict[str, Any]) -> bool:
    """Validate that chunk has required fields."""
    required = ["id", "contents"]
    return all(field in chunk for field in required) and isinstance(chunk["contents"], str)


def _parse_cuda_device(requested: str) -> Optional[int]:
    """Extract CUDA device index from string like 'cuda' or 'cuda:1'."""
    if torch is None or not torch.cuda.is_available():
        return None
    if requested == "cuda":
        try:
            return torch.cuda.current_device()
        except Exception:
            return 0
    if ":" in requested:
        try:
            return int(requested.split(":", 1)[1])
        except ValueError:
            return None
    return None


def resolve_device(preferred: Optional[str] = None, fallback: str = "cpu") -> str:
    """
    Resolve the best available device given a user preference.

    - "auto": pick CUDA if available, else CPU.
    - "cuda" or "cuda:{id}": verify CUDA availability, otherwise fall back.
    - "cpu": always CPU.
    """
    pref = (preferred or "auto").lower()
    if pref in {"cpu", "cpu:0"}:
        return "cpu"

    if torch is None:
        raise GPUMisconfigurationError("PyTorch is not available; cannot resolve a CUDA device.")

    if pref == "auto":
        return get_device()

    if pref.startswith("cuda"):
        base_device = get_device()
        if ":" not in pref:
            return base_device
        idx = _parse_cuda_device(pref)
        if idx is None:
            return base_device
        if idx >= torch.cuda.device_count():  # type: ignore[arg-type]
            raise GPUMisconfigurationError(
                f"Requested CUDA device cuda:{idx} but only {torch.cuda.device_count()} device(s) detected."
            )
        ensure_cuda_device(idx)
        return f"cuda:{idx}"

    return fallback


def supports_fp16(device: str) -> bool:
    """Return True if the runtime supports fp16 on the specified device."""
    if torch is None:
        return False
    if not device.startswith("cuda"):
        return False
    try:
        idx = _parse_cuda_device(device) or 0
        major, minor = torch.cuda.get_device_capability(idx)
        return major >= 6  # Pascal+ have native fp16 support
    except Exception:  # pragma: no cover - depends on hardware
        return False

