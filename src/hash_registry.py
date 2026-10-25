"""
Hash Registry — content-based deduplication for incremental FAISS updates.

Stores a JSON file that maps SHA-256 hashes of chunk text to a compact
record of {metadata, text_preview}.  Embeddings are NOT stored here —
they live only in the FAISS index, which keeps this file tiny even at
hundreds of thousands of chunks.

Storage layout (data/hash_registry.json):
{
    "<sha256-hex>": {
        "metadata": {"act_name": "...", "page": 0, "section": "..."},
        "text": "first 200 chars...",
        "last_updated": "2025-01-01T00:00:00+00:00"
    },
    ...
}
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Set


class HashRegistry:
    """Persistent SHA-256 → chunk-metadata cache backed by a JSON file."""

    def __init__(self, registry_path: Path):
        self._path = Path(registry_path)
        self._data: Dict[str, Dict[str, Any]] = {}

    # ── persistence ──────────────────────────────────────────────

    def load(self) -> "HashRegistry":
        if self._path.exists():
            with open(self._path, "r", encoding="utf-8") as f:
                self._data = json.load(f)
        return self

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False)

    # ── CRUD ─────────────────────────────────────────────────────

    def get(self, chunk_hash: str) -> Optional[Dict[str, Any]]:
        return self._data.get(chunk_hash)

    def put(
        self,
        chunk_hash: str,
        metadata: Dict[str, Any],
        text_preview: str = "",
    ) -> None:
        """Store (or overwrite) an entry. No embedding is stored."""
        self._data[chunk_hash] = {
            "metadata": metadata,
            "text": text_preview[:200],
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }

    def contains(self, chunk_hash: str) -> bool:
        return chunk_hash in self._data

    def prune(self, active_hashes: Set[str]) -> int:
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
        return hashlib.sha256(text.encode("utf-8")).hexdigest()