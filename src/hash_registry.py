"""
Hash Registry — content-based deduplication for incremental FAISS updates.

Maintains a JSON file that maps SHA-256 hashes of chunk text to their
pre-computed embeddings and metadata.  This lets the auto-ingest pipeline
skip the expensive embedding step for chunks that haven't changed.

Storage layout (data/hash_registry.json):
{
    "<sha256-hex>": {
        "embedding": [0.012, -0.034, ...],
        "metadata": {"source": "...", "page": 0, "last_updated": "..."},
        "text": "first 120 chars..."
    },
    ...
}
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


class HashRegistry:
    """Persistent SHA-256 → embedding cache backed by a JSON file."""

    def __init__(self, registry_path: Path):
        self._path = Path(registry_path)
        self._data: Dict[str, Dict[str, Any]] = {}

    # ── persistence ──────────────────────────────────────────────

    def load(self) -> "HashRegistry":
        """Load the registry from disk.  No-op if the file doesn't exist."""
        if self._path.exists():
            with open(self._path, "r", encoding="utf-8") as f:
                self._data = json.load(f)
        return self

    def save(self) -> None:
        """Persist the current registry to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False)

    # ── CRUD ─────────────────────────────────────────────────────

    def get(self, chunk_hash: str) -> Optional[Dict[str, Any]]:
        """Return cached entry for *chunk_hash*, or ``None``."""
        return self._data.get(chunk_hash)

    def put(
        self,
        chunk_hash: str,
        embedding: List[float],
        metadata: Dict[str, Any],
        text_preview: str = "",
    ) -> None:
        """Store (or overwrite) an entry."""
        self._data[chunk_hash] = {
            "embedding": embedding,
            "metadata": {
                **metadata,
                "last_updated": datetime.now(timezone.utc).isoformat(),
            },
            "text": text_preview[:120],
        }

    def contains(self, chunk_hash: str) -> bool:
        return chunk_hash in self._data

    def prune(self, active_hashes: Set[str]) -> int:
        """Remove entries whose hash is NOT in *active_hashes*.

        Returns the number of entries removed.
        """
        stale = set(self._data.keys()) - active_hashes
        for h in stale:
            del self._data[h]
        return len(stale)

    @property
    def size(self) -> int:
        return len(self._data)

    def all_hashes(self) -> Set[str]:
        return set(self._data.keys())

    # ── hashing helper ───────────────────────────────────────────

    @staticmethod
    def compute_hash(text: str) -> str:
        """SHA-256 hex digest of the given text."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
