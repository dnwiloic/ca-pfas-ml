"""
Cache persistant pour la collecte Phase 1.

- ``data/cache/index.json`` : état par source (taille, sha256, URL, statut)
- ``data/cache/ckan_urls.json`` : URLs CKAN résolues (évite l'API resource_show)
- Reprise des téléchargements via ``*.part`` + ``*.part.meta``
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CHUNK = 1 << 20
MIN_VALID_BYTES = 1024  # rejette les pages d'erreur HTML minuscules


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def build_ckan_download_url(resource_id: str, filename: str, package_uuid: str) -> str:
    """URL stable data.ca.gov — pas d'appel API CKAN nécessaire."""
    return (
        f"https://data.ca.gov/dataset/{package_uuid}/resource/"
        f"{resource_id}/download/{filename}"
    )


class DownloadCache:
    """Index local : ne retélécharge pas si le fichier cible est déjà valide."""

    def __init__(self, cache_dir: Path, project_root: Path) -> None:
        self.cache_dir = cache_dir
        self.project_root = project_root
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = cache_dir / "index.json"
        self.ckan_path = cache_dir / "ckan_urls.json"
        self._index: dict[str, Any] = self._load_json(self.index_path, {"version": 1, "sources": {}})
        self._ckan_urls: dict[str, str] = self._load_json(self.ckan_path, {})

    @staticmethod
    def _load_json(path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Cache corrompu %s — réinitialisation", path)
            return default

    def _save_index(self) -> None:
        self.index_path.write_text(
            json.dumps(self._index, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _save_ckan_urls(self) -> None:
        self.ckan_path.write_text(
            json.dumps(self._ckan_urls, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def get_entry(self, source_id: str) -> dict[str, Any]:
        return self._index.setdefault("sources", {}).get(source_id, {})

    def get_ckan_url(self, resource_id: str) -> str | None:
        return self._ckan_urls.get(resource_id)

    def set_ckan_url(self, resource_id: str, url: str) -> None:
        if self._ckan_urls.get(resource_id) == url:
            return
        self._ckan_urls[resource_id] = url
        self._save_ckan_urls()

    def dest_path(self, dest_rel: str) -> Path:
        return self.project_root / "data" / "raw" / dest_rel

    def part_paths(self, dest: Path) -> tuple[Path, Path]:
        return dest.with_suffix(dest.suffix + ".part"), dest.with_suffix(dest.suffix + ".part.meta")

    def is_valid_file(self, path: Path, *, min_bytes: int = MIN_VALID_BYTES) -> bool:
        if not path.is_file():
            return False
        try:
            return path.stat().st_size >= min_bytes
        except OSError:
            return False

    def is_complete(
        self,
        source_id: str,
        dest_rel: str,
        *,
        force: bool = False,
        url: str | None = None,
    ) -> tuple[bool, str]:
        """
        Retourne (True, raison) si le fichier final peut être réutilisé sans réseau.
        """
        if force:
            return False, "force"

        dest = self.dest_path(dest_rel)
        if not self.is_valid_file(dest):
            return False, "absent ou trop petit"

        entry = self.get_entry(source_id)
        if entry.get("status") == "ok" and entry.get("dest") == dest_rel:
            stored = entry.get("bytes")
            if stored and abs(dest.stat().st_size - int(stored)) > 64:
                return False, "taille différente de l'index"
            if url and entry.get("url") and entry["url"] != url:
                return False, "URL changée"
            return True, "cache index"

        # Fichier présent sur disque mais pas encore indexé → indexer et réutiliser
        if self.is_valid_file(dest):
            return True, "fichier disque (non indexé)"

        return False, "incomplet"

    def mark_ok(
        self,
        source_id: str,
        dest_rel: str,
        *,
        url: str | None = None,
        bytes_size: int | None = None,
        sha256: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        dest = self.dest_path(dest_rel)
        size = bytes_size if bytes_size is not None else dest.stat().st_size
        entry: dict[str, Any] = {
            "dest": dest_rel,
            "status": "ok",
            "bytes": size,
            "updated_at": _utc_now(),
        }
        if url:
            entry["url"] = url
        if sha256:
            entry["sha256"] = sha256
        elif dest.is_file() and size < 50 * 1024 * 1024:
            entry["sha256"] = _sha256_file(dest)
        if extra:
            entry.update(extra)
        self._index.setdefault("sources", {})[source_id] = entry
        self._save_index()
        self._clear_part(dest)

    def mark_error(self, source_id: str, dest_rel: str, error: str, url: str | None = None) -> None:
        entry = {
            "dest": dest_rel,
            "status": "error",
            "error": error[:500],
            "updated_at": _utc_now(),
        }
        if url:
            entry["url"] = url
        self._index.setdefault("sources", {})[source_id] = entry
        self._save_index()

    def _clear_part(self, dest: Path) -> None:
        part, meta = self.part_paths(dest)
        for p in (part, meta):
            if p.exists():
                p.unlink()

    def save_part_meta(self, dest: Path, *, source_id: str, url: str, bytes_done: int) -> None:
        _, meta = self.part_paths(dest)
        meta.write_text(
            json.dumps(
                {
                    "source_id": source_id,
                    "url": url,
                    "bytes": bytes_done,
                    "updated_at": _utc_now(),
                }
            ),
            encoding="utf-8",
        )

    def load_part_meta(self, dest: Path) -> dict[str, Any] | None:
        _, meta = self.part_paths(dest)
        if not meta.exists():
            return None
        try:
            return json.loads(meta.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def part_resume_bytes(self, dest: Path, url: str, source_id: str) -> int:
        """Octets déjà téléchargés si reprise compatible."""
        part, _ = self.part_paths(dest)
        if not part.exists():
            return 0
        meta = self.load_part_meta(dest)
        if meta and (meta.get("url") != url or meta.get("source_id") != source_id):
            logger.warning(
                "Partiel %s : URL/source différente — redémarrage à zéro",
                part.name,
            )
            part.unlink(missing_ok=True)
            return 0
        return part.stat().st_size

    def sync_from_disk(self, sources: list[Any]) -> int:
        """Scanne data/raw et met à jour l'index pour les fichiers déjà présents."""
        n = 0
        for spec in sources:
            if spec.kind in ("manual",):
                continue
            dest_rel = spec.dest
            if dest_rel.endswith("/"):
                continue
            dest = self.dest_path(dest_rel)
            if not self.is_valid_file(dest):
                continue
            cached, _ = self.is_complete(spec.id, dest_rel, force=False)
            if cached and self.get_entry(spec.id).get("status") == "ok":
                continue
            url = spec.url or None
            if spec.kind == "ckan" and spec.resource_id:
                fn = Path(dest_rel).name
                try:
                    from .sources_registry import PACKAGE_UUID
                except ImportError:
                    from sources_registry import PACKAGE_UUID
                url = build_ckan_download_url(spec.resource_id, fn, PACKAGE_UUID)
            self.mark_ok(spec.id, dest_rel, url=url)
            n += 1
            logger.info("[sync] %s indexé (%s)", spec.id, _human_size(dest.stat().st_size))
        return n

    def failed_ids(self) -> list[str]:
        return [
            sid
            for sid, e in self._index.get("sources", {}).items()
            if e.get("status") == "error"
        ]

    def arcgis_checkpoint_path(self, source_id: str) -> Path:
        d = self.cache_dir / "arcgis"
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{source_id}.json"

    def load_arcgis_checkpoint(self, source_id: str) -> dict[str, Any] | None:
        p = self.arcgis_checkpoint_path(source_id)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def save_arcgis_checkpoint(
        self,
        source_id: str,
        *,
        offset: int,
        n_features: int,
        use_geojson: bool,
    ) -> None:
        p = self.arcgis_checkpoint_path(source_id)
        p.write_text(
            json.dumps(
                {
                    "offset": offset,
                    "n_features": n_features,
                    "use_geojson": use_geojson,
                    "updated_at": _utc_now(),
                }
            ),
            encoding="utf-8",
        )

    def clear_arcgis_checkpoint(self, source_id: str) -> None:
        p = self.arcgis_checkpoint_path(source_id)
        if p.exists():
            p.unlink()


def _human_size(n: int) -> str:
    for u in ("o", "Ko", "Mo", "Go"):
        if n < 1024:
            return f"{n:.0f} {u}" if u == "o" else f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} To"
