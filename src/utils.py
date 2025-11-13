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


def _try_cuda_device(device_str: str) -> bool:
    """Return True if a tiny allocation/sync on the device succeeds."""
    if torch is None:
        return False
    try:
        idx = _parse_cuda_device(device_str)
        target = device_str if ":" in device_str else (f"cuda:{idx}" if idx is not None else "cuda")
        with torch.cuda.device(idx if idx is not None else target):
            torch.zeros(1, device=target)
            torch.cuda.synchronize()
        return True
    except Exception as exc:  # pragma: no cover - depends on hardware
        logger.warning("CUDA device %s unavailable (%s); falling back to CPU", device_str, exc)
        return False


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

    if torch is None or not torch.cuda.is_available():
        return fallback

    # handle auto selection
    if pref == "auto":
        default_device = "cuda"
        if _try_cuda_device(default_device):
            idx = _parse_cuda_device(default_device)
            return f"cuda:{idx}" if idx is not None else "cuda"
        return fallback

    if pref.startswith("cuda"):
        idx = _parse_cuda_device(pref)
        device_str = pref if idx is not None else "cuda"
        if idx is not None and idx >= torch.cuda.device_count():  # type: ignore[arg-type]
            logger.warning(
                "Requested CUDA device %s outside available range (count=%s); falling back to CPU",
                idx,
                torch.cuda.device_count(),
            )
            return fallback
        if _try_cuda_device(device_str):
            return device_str if ":" in device_str else f"cuda:{_parse_cuda_device(device_str) or 0}"
        return fallback

    # default fallback
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

