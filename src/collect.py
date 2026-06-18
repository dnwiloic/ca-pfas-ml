"""
Phase 1 — Collecte des 23 sources (CA-PFAS-ASGWS, Dong et al. 2024).

Cache : ``data/cache/index.json`` — ne retélécharge pas les fichiers déjà valides.
Reprise : ``*.part`` + métadonnées si coupure réseau.

Usage :
    python -m src.collect --list
    python -m src.collect --group gama --max-mb 0
    python -m src.collect --sync-cache          # indexer les fichiers déjà sur disque
    python -m src.collect --retry-errors --group gama
    python -m src.collect --status
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from .download_cache import (
        DownloadCache,
        build_ckan_download_url,
    )
    from .sources_registry import (
        CKAN_API,
        PACKAGE_UUID,
        SOURCE_BY_ID,
        SOURCES,
        SourceSpec,
    )
except ImportError:
    from download_cache import DownloadCache, build_ckan_download_url
    from sources_registry import CKAN_API, PACKAGE_UUID, SOURCE_BY_ID, SOURCES, SourceSpec

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
MANIFEST_PATH = RAW_DIR / "manifest.json"

USER_AGENT = "CA-PFAS-ASGWS-collect/1.0 (research; +https://data.ca.gov)"
CHUNK = 1 << 20
CONNECT_TIMEOUT = 30
READ_TIMEOUT = 900  # gros CSV S3 (plusieurs Go)
CKAN_API_TIMEOUT = 120
ARCGIS_TIMEOUT = 300


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=["GET", "HEAD"],
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def _skipped_result(
    dcache: DownloadCache,
    source_id: str,
    dest_rel: str,
    reason: str,
) -> dict[str, Any]:
    dest = dcache.dest_path(dest_rel)
    entry = dcache.get_entry(source_id)
    logger.info(
        "[cache] %s — %s (%s)",
        source_id,
        reason,
        _human_size(dest.stat().st_size),
    )
    return {
        "status": "skipped",
        "path": str(dest.relative_to(PROJECT_ROOT)),
        "bytes": dest.stat().st_size,
        "sha256": entry.get("sha256"),
        "cache_reason": reason,
    }


def _human_size(n: int) -> str:
    for u in ("o", "Ko", "Mo", "Go"):
        if n < 1024:
            return f"{n:.0f} {u}" if u == "o" else f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} To"


def download_stream(
    session: requests.Session,
    dcache: DownloadCache,
    source_id: str,
    url: str,
    dest_rel: str,
    *,
    force: bool = False,
    max_mb: float | None = None,
) -> dict[str, Any]:
    """Télécharge avec cache, reprise Range et retries explicites."""
    dest = dcache.dest_path(dest_rel)
    dest.parent.mkdir(parents=True, exist_ok=True)

    complete, reason = dcache.is_complete(source_id, dest_rel, force=force, url=url)
    if complete:
        return _skipped_result(dcache, source_id, dest_rel, reason)

    part, _ = dcache.part_paths(dest)
    existing = 0 if force else dcache.part_resume_bytes(dest, url, source_id)
    headers: dict[str, str] = {}
    mode = "wb"
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"
        logger.info("[reprise] %s depuis %s", dest.name, _human_size(existing))

    logger.info("GET %s → %s", url[:90], dest.name)
    last_err: Exception | None = None
    for attempt in range(6):
        try:
            with session.get(
                url,
                stream=True,
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
                headers=headers,
            ) as resp:
                if resp.status_code == 416 and part.exists():
                    part.rename(dest)
                    break
                resp.raise_for_status()
                if max_mb is not None:
                    cl = resp.headers.get("Content-Length")
                    total = (int(cl) + existing) if cl and cl.isdigit() else None
                    if total and total > max_mb * 1024 * 1024:
                        raise RuntimeError(
                            f"Fichier > {max_mb} Mo (≈ {total / 1e6:.0f} Mo). "
                            "Augmentez --max-mb."
                        )
                with part.open(mode) as f:
                    downloaded = existing
                    for chunk in resp.iter_content(chunk_size=CHUNK):
                        if not chunk:
                            continue
                        f.write(chunk)
                        downloaded += len(chunk)
                        if downloaded % (50 * 1024 * 1024) < CHUNK:
                            dcache.save_part_meta(dest, source_id=source_id, url=url, bytes_done=downloaded)
            if part.exists():
                part.rename(dest)
            if not dcache.is_valid_file(dest):
                raise RuntimeError(f"Téléchargement incomplet : {dest}")
            dcache.mark_ok(source_id, dest_rel, url=url)
            return {
                "status": "ok",
                "path": str(dest.relative_to(PROJECT_ROOT)),
                "bytes": dest.stat().st_size,
                "sha256": dcache.get_entry(source_id).get("sha256"),
                "resumed": existing > 0,
            }
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            last_err = exc
            wait = min(60, 2 ** attempt)
            logger.warning(
                "Tentative %d/6 %s : %s — pause %ds (partiel %s)",
                attempt + 1,
                source_id,
                exc,
                wait,
                _human_size(part.stat().st_size) if part.exists() else "0",
            )
            time.sleep(wait)
            existing = dcache.part_resume_bytes(dest, url, source_id)
            headers = {"Range": f"bytes={existing}-"} if existing else {}
            mode = "ab" if existing else "wb"

    dcache.mark_error(source_id, dest_rel, str(last_err), url=url)
    raise last_err or RuntimeError("échec téléchargement")


def resolve_ckan_url(
    session: requests.Session,
    dcache: DownloadCache,
    spec: SourceSpec,
) -> str:
    """URL CKAN sans appel API si possible."""
    filename = Path(spec.dest).name
    url = build_ckan_download_url(spec.resource_id, filename, PACKAGE_UUID)
    cached = dcache.get_ckan_url(spec.resource_id)
    if cached == url:
        return url

    dcache.set_ckan_url(spec.resource_id, url)

    # Fallback API uniquement si le fichier local est absent et URL incertaine
    if dcache.is_valid_file(dcache.dest_path(spec.dest)):
        return url

    try:
        r = session.get(
            f"{CKAN_API}/resource_show",
            params={"id": spec.resource_id},
            timeout=CKAN_API_TIMEOUT,
        )
        r.raise_for_status()
        api_url = r.json()["result"].get("url", "")
        if api_url and "/download/" in api_url:
            dcache.set_ckan_url(spec.resource_id, api_url)
            return api_url
    except requests.RequestException as exc:
        logger.debug("resource_show ignoré (%s), URL construite", exc)

    return url


def collect_ckan(
    session: requests.Session,
    dcache: DownloadCache,
    spec: SourceSpec,
    *,
    force: bool,
    max_mb: float | None,
) -> dict[str, Any]:
    url = resolve_ckan_url(session, dcache, spec)
    return download_stream(
        session, dcache, spec.id, url, spec.dest, force=force, max_mb=max_mb
    )


def collect_url(
    session: requests.Session,
    dcache: DownloadCache,
    spec: SourceSpec,
    *,
    force: bool,
    max_mb: float | None,
) -> dict[str, Any]:
    return download_stream(
        session, dcache, spec.id, spec.url, spec.dest, force=force, max_mb=max_mb
    )


def collect_arcgis(
    session: requests.Session,
    dcache: DownloadCache,
    spec: SourceSpec,
    *,
    force: bool,
    page_size: int | None = None,
    max_retries: int = 5,
) -> dict[str, Any]:
    dest_rel = spec.dest
    dest = dcache.dest_path(dest_rel)

    complete, reason = dcache.is_complete(spec.id, dest_rel, force=force)
    if complete:
        return _skipped_result(dcache, spec.id, dest_rel, reason)

    dest.parent.mkdir(parents=True, exist_ok=True)
    use_geojson = dest.suffix.lower() == ".geojson"
    if page_size is None:
        page_size = 100 if "gis.water.ca.gov" in spec.arcgis_query_url else 1000

    all_features: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    offset = 0

    base = spec.arcgis_query_url
    sgma_fields = (
        "Basin_Number,Basin_Subbasin_Number,Basin_Name,"
        "Basin_Subbasin_Name,Region_Office,Area_SqMiles"
    )
    is_dwr_sgma = "i08_B118_CA_GroundwaterBasins" in spec.arcgis_query_url

    while True:
        params: dict[str, Any] = {
            "where": "1=1",
            "outFields": sgma_fields if is_dwr_sgma else "*",
            "returnGeometry": "true",
            "outSR": "4326",
            "resultOffset": offset,
            "resultRecordCount": page_size,
        }
        if is_dwr_sgma:
            params["orderByFields"] = "Basin_Number"
        params["f"] = "geojson" if use_geojson else "json"

        data = None
        for attempt in range(max_retries):
            try:
                resp = session.get(
                    base, params=params, timeout=(CONNECT_TIMEOUT, ARCGIS_TIMEOUT)
                )
                resp.raise_for_status()
                data = resp.json()
                break
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                if attempt + 1 >= max_retries:
                    raise
                wait = 2 ** attempt
                logger.warning("ArcGIS %s offset=%d : %s (retry %ds)", spec.id, offset, exc, wait)
                time.sleep(wait)

        assert data is not None
        features = data.get("features", [])
        if use_geojson:
            all_features.extend(features)
        else:
            for feat in features:
                attrs = dict(feat.get("attributes") or {})
                geom = feat.get("geometry") or {}
                lon = geom.get("x") or attrs.get("longitude") or attrs.get("LONGITUDE")
                lat = (
                    geom.get("y")
                    or attrs.get("latitude")
                    or attrs.get("latitiude")
                    or attrs.get("LATITUDE")
                )
                if lat is not None and lon is not None:
                    attrs["latitude"] = lat
                    attrs["longitude"] = lon
                rows.append(attrs)

        logger.info("  %s offset=%d +%d", spec.id, offset, len(features))
        offset += len(features)
        dcache.save_arcgis_checkpoint(
            spec.id, offset=offset, n_features=len(all_features) or len(rows), use_geojson=use_geojson
        )

        if len(features) < page_size:
            break
        time.sleep(0.3)

    if use_geojson:
        collection: dict[str, Any] = {"type": "FeatureCollection", "features": all_features}
        dest.write_text(json.dumps(collection), encoding="utf-8")
        n_rows = len(all_features)
    else:
        import pandas as pd

        pd.DataFrame(rows).to_csv(dest, index=False)
        n_rows = len(rows)

    dcache.mark_ok(spec.id, dest_rel, extra={"n_rows": n_rows})
    dcache.clear_arcgis_checkpoint(spec.id)
    return {
        "status": "ok",
        "path": str(dest.relative_to(PROJECT_ROOT)),
        "bytes": dest.stat().st_size,
        "n_rows": n_rows,
    }


def collect_local(dcache: DownloadCache, spec: SourceSpec, *, force: bool) -> dict[str, Any]:
    dest_dir = dcache.dest_path(spec.dest)
    src_dir = PROJECT_ROOT / Path(spec.local_glob).parent
    glob_pat = Path(spec.local_glob).name
    matches = sorted(src_dir.glob(glob_pat))

    if not matches:
        dcache.mark_error(spec.id, spec.dest, f"fichiers absents: {spec.local_glob}")
        return {
            "status": "missing",
            "path": str(dest_dir.relative_to(PROJECT_ROOT)),
            "notes": f"Télécharger via {spec.url}",
        }

    dest_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    total = 0
    for src in matches:
        dst = dest_dir / src.name
        if dst.exists() and not force:
            mtime_dst, mtime_src = dst.stat().st_mtime, src.stat().st_mtime
            if dst.stat().st_size == src.stat().st_size and mtime_dst >= mtime_src:
                copied.append(dst.name)
                total += dst.stat().st_size
                continue
        shutil.copy2(src, dst)
        copied.append(dst.name)
        total += dst.stat().st_size

    dcache.mark_ok(spec.id, spec.dest, extra={"files": copied})
    return {
        "status": "ok",
        "path": str(dest_dir.relative_to(PROJECT_ROOT)),
        "files": copied,
        "bytes": total,
    }


def collect_manual(spec: SourceSpec) -> dict[str, Any]:
    dest = RAW_DIR / spec.dest
    if dest.suffix:
        readme = dest.parent / f"README_{dest.stem}.txt"
        dest.parent.mkdir(parents=True, exist_ok=True)
    else:
        dest.mkdir(parents=True, exist_ok=True)
        readme = dest / "README.txt"
    if readme.exists():
        return {
            "status": "skipped",
            "path": str(dest.relative_to(PROJECT_ROOT)),
            "readme": str(readme.relative_to(PROJECT_ROOT)),
        }
    readme.write_text(
        f"# {spec.name}\n\nURL : {spec.url}\n\n{spec.notes}\n",
        encoding="utf-8",
    )
    return {
        "status": "manual",
        "path": str(dest.relative_to(PROJECT_ROOT)),
        "readme": str(readme.relative_to(PROJECT_ROOT)),
    }


def _sgma_cache_fallback(dcache: DownloadCache, spec: SourceSpec, force: bool) -> dict[str, Any] | None:
    cache = PROJECT_ROOT.parent / "usgs2_GAMA_PBP" / "data" / "cache" / "sgma_basins.geojson"
    if not cache.exists():
        return None
    dest_rel = spec.dest
    complete, reason = dcache.is_complete(spec.id, dest_rel, force=force)
    if complete:
        return _skipped_result(dcache, spec.id, dest_rel, reason)
    shutil.copy2(cache, dcache.dest_path(dest_rel))
    dcache.mark_ok(spec.id, dest_rel, extra={"from_cache": str(cache)})
    return {
        "status": "ok",
        "path": dest_rel,
        "bytes": dcache.dest_path(dest_rel).stat().st_size,
        "from_cache": str(cache),
    }


def collect_one(
    session: requests.Session,
    dcache: DownloadCache,
    spec: SourceSpec,
    *,
    force: bool,
    max_mb: float | None,
    skip_manual: bool,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "id": spec.id,
        "name": spec.name,
        "category": spec.category,
        "kind": spec.kind,
        "dest": spec.dest,
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        if spec.kind == "manual":
            if skip_manual:
                entry["status"] = "skipped_manual"
                return entry
            entry.update(collect_manual(spec))
            return entry
        if spec.kind == "ckan":
            entry.update(collect_ckan(session, dcache, spec, force=force, max_mb=max_mb))
        elif spec.kind == "url":
            entry.update(collect_url(session, dcache, spec, force=force, max_mb=max_mb))
        elif spec.kind == "arcgis":
            try:
                entry.update(collect_arcgis(session, dcache, spec, force=force))
            except Exception:
                if spec.id == "sgma_groundwater_basins":
                    fb = _sgma_cache_fallback(dcache, spec, force)
                    if fb:
                        entry.update(fb)
                        return entry
                raise
        elif spec.kind == "local":
            entry.update(collect_local(dcache, spec, force=force))
        else:
            entry["status"] = "error"
            entry["error"] = f"kind inconnu: {spec.kind}"
    except Exception as exc:
        logger.error("Échec %s : %s", spec.id, exc)
        entry["status"] = "error"
        entry["error"] = str(exc)[:500]
    return entry


def load_manifest() -> dict[str, Any]:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {"version": 1, "runs": []}


def save_manifest(manifest: dict[str, Any]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def print_status(dcache: DownloadCache) -> None:
    print(f"{'ID':<32} {'STAT':<10} {'TAILLE':>10}  FICHIER")
    print("-" * 75)
    for spec in SOURCES:
        dest = dcache.dest_path(spec.dest) if not spec.dest.endswith("/") else None
        entry = dcache.get_entry(spec.id)
        st = entry.get("status", "—")
        if dest and dcache.is_valid_file(dest):
            size = _human_size(dest.stat().st_size)
            part, _ = dcache.part_paths(dest)
            if part.exists():
                st = "partiel"
                size = f"{size} + {_human_size(part.stat().st_size)} part"
        elif entry.get("status") == "error":
            size = "erreur"
        else:
            size = "—"
        print(f"{spec.id:<32} {st:<10} {size:>10}  {spec.dest}")


def run_collect(
    specs: list[SourceSpec],
    *,
    force: bool = False,
    max_mb: float | None = 200.0,
    include_optional: bool = False,
    skip_manual: bool = False,
    sync_first: bool = False,
) -> dict[str, Any]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dcache = DownloadCache(CACHE_DIR, PROJECT_ROOT)
    if sync_first:
        n = dcache.sync_from_disk(SOURCES)
        logger.info("Synchronisation cache : %d entrée(s) mise(s) à jour", n)

    session = _session()
    results: list[dict[str, Any]] = []

    for spec in specs:
        if spec.optional and not include_optional:
            results.append({"id": spec.id, "status": "skipped_optional"})
            continue
        logger.info("=== %s (%s) ===", spec.id, spec.kind)
        results.append(
            collect_one(
                session,
                dcache,
                spec,
                force=force,
                max_mb=max_mb,
                skip_manual=skip_manual,
            )
        )

    summary = {
        "ok": sum(1 for r in results if r.get("status") == "ok"),
        "skipped": sum(1 for r in results if r.get("status", "").startswith("skipped")),
        "manual": sum(1 for r in results if r.get("status") == "manual"),
        "missing": sum(1 for r in results if r.get("status") == "missing"),
        "error": sum(1 for r in results if r.get("status") == "error"),
    }

    manifest = load_manifest()
    run = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "n_sources": len(specs),
        "summary": summary,
        "results": results,
    }
    manifest.setdefault("runs", []).append(run)
    manifest["last_run"] = run
    save_manifest(manifest)

    logger.info(
        "Terminé : %d ok, %d cache/ignorés, %d manuels, %d manquants, %d erreurs",
        summary["ok"],
        summary["skipped"],
        summary["manual"],
        summary["missing"],
        summary["error"],
    )
    return run


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1 — collecte CA-PFAS-ASGWS (cache)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--list", action="store_true")
    g.add_argument("--all", action="store_true")
    g.add_argument("--status", action="store_true", help="État cache / fichiers")
    g.add_argument("--sync-cache", action="store_true", help="Indexer data/raw sans télécharger")
    p.add_argument("--group", choices=["gama", "contamination", "environment"], action="append")
    p.add_argument("--only", action="append")
    p.add_argument("--force", action="store_true", help="Ignorer le cache et retélécharger")
    p.add_argument("--retry-errors", action="store_true", help="Relancer uniquement les sources en erreur")
    p.add_argument("--max-mb", type=float, default=200.0, help="0 = illimité")
    p.add_argument("--include-optional", action="store_true")
    p.add_argument("--fetch-manual-readmes", action="store_true")
    return p.parse_args(argv)


def select_specs(args: argparse.Namespace, dcache: DownloadCache | None = None) -> list[SourceSpec]:
    if args.retry_errors and dcache:
        ids = set(dcache.failed_ids())
        specs = [s for s in SOURCES if s.id in ids]
        if specs:
            return specs
        logger.warning("Aucune source en erreur dans le cache — groupe par défaut")
    if args.only:
        return [SOURCE_BY_ID[i] for i in args.only]
    if args.group:
        cats = set(args.group)
        return [s for s in SOURCES if s.category in cats]
    if args.all:
        return list(SOURCES)
    return [s for s in SOURCES if s.category == "gama"]


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    dcache = DownloadCache(CACHE_DIR, PROJECT_ROOT)

    if args.list:
        print(f"{'ID':<32} {'CAT':<14} {'KIND':<8} {'OPT':<4} DEST")
        print("-" * 90)
        for s in SOURCES:
            opt = "yes" if s.optional else ""
            print(f"{s.id:<32} {s.category:<14} {s.kind:<8} {opt:<4} {s.dest}")
        print(f"\nTotal : {len(SOURCES)} sources")
        return 0

    if args.status:
        print_status(dcache)
        return 0

    if args.sync_cache:
        n = dcache.sync_from_disk(SOURCES)
        print(f"Index mis à jour : {n} source(s). Voir {CACHE_DIR / 'index.json'}")
        return 0

    specs = select_specs(args, dcache)
    if not specs:
        logger.error("Aucune source sélectionnée.")
        return 1

    max_mb = None if args.max_mb <= 0 else args.max_mb
    skip_manual = not args.fetch_manual_readmes and not args.include_optional

    run_collect(
        specs,
        force=args.force,
        max_mb=max_mb,
        include_optional=args.include_optional,
        skip_manual=skip_manual,
        sync_first=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
