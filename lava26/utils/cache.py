import hashlib
import pickle
import shutil
from pathlib import Path
from typing import Any, Optional

import torch


class DiskCache:
    def __init__(self, cache_dir: str):
        self.root = Path(cache_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def _key_path(self, key: str, ext: str) -> Path:
        h = hashlib.sha256(key.encode()).hexdigest()[:16]
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)[:64]
        return self.root / f"{safe}_{h}{ext}"

    def save(self, key: str, data: Any) -> None:
        if isinstance(data, torch.Tensor):
            path = self._key_path(key, ".pt")
            torch.save(data, path)
        else:
            path = self._key_path(key, ".pkl")
            with open(path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, key: str) -> Optional[Any]:
        for ext, loader in [(".pt", torch.load), (".pkl", None)]:
            path = self._key_path(key, ext)
            if path.exists():
                if ext == ".pt":
                    return loader(path, map_location="cpu", weights_only=False)
                else:
                    with open(path, "rb") as f:
                        return pickle.load(f)
        return None

    def exists(self, key: str) -> bool:
        return any(
            self._key_path(key, ext).exists() for ext in (".pt", ".pkl")
        )

    def clear(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
