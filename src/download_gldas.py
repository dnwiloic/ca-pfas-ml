"""
Téléchargement des fichiers GLDAS NOAH025 Monthly v2.1 depuis NASA Earthdata.

Lit les URLs depuis le fichier .txt généré par GES DISC Subset Tool.
Skip les fichiers déjà présents et complets.
Authentification via NASA_EARTHDATA_USER / NASA_EARTHDATA_PASSWORD dans .env

Usage :
    python -m src.download_gldas
    python -m src.download_gldas --workers 4
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GLDAS_DIR = PROJECT_ROOT / "data" / "raw" / "environment" / "nasa_gldas"
URL_LIST = GLDAS_DIR / "subset_GLDAS_NOAH025_M_2.1_20260610_030530_.txt"
EARTHDATA_AUTH_URL = "https://urs.earthdata.nasa.gov"
MIN_FILE_BYTES = 100_000   # fichier valide > 100 Ko


def _load_env(name: str) -> str:
    val = os.environ.get(name, "")
    if val:
        return val
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.strip().startswith(f"{name}="):
                return line.split("=", 1)[1].strip()
    return ""


def _make_session(user: str, password: str, token: str = "") -> requests.Session:
    session = requests.Session()
    retry = Retry(total=5, backoff_factor=1,
                  status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    if token:
        # JWT Bearer token (préféré — pas de redirect loop)
        session.headers["Authorization"] = f"Bearer {token}"
    else:
        # Basic Auth avec suivi des redirections Earthdata
        session.auth = (user, password)
    return session


def _is_complete(path: Path) -> bool:
    return path.is_file() and path.stat().st_size >= MIN_FILE_BYTES


def _download_one(session: requests.Session, url: str, dest: Path) -> tuple[str, bool, str]:
    """Télécharge un fichier. Retourne (filename, success, message)."""
    fname = dest.name
    if _is_complete(dest):
        return fname, True, "skip (déjà présent)"

    # Reprise partielle
    headers: dict[str, str] = {}
    part = dest.with_suffix(dest.suffix + ".part")
    resume_pos = part.stat().st_size if part.exists() else 0
    if resume_pos > 0:
        headers["Range"] = f"bytes={resume_pos}-"

    try:
        r = session.get(url, headers=headers, stream=True, timeout=120)
        if r.status_code == 401:
            return fname, False, "401 — credentials incorrects"
        if r.status_code == 404:
            return fname, False, "404 — fichier absent sur le serveur"
        r.raise_for_status()

        mode = "ab" if resume_pos > 0 and r.status_code == 206 else "wb"
        with open(part, mode) as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)

        if not _is_complete(part):
            return fname, False, f"fichier trop petit ({part.stat().st_size} o)"

        part.rename(dest)
        return fname, True, f"OK ({dest.stat().st_size // 1024} Ko)"

    except requests.RequestException as e:
        return fname, False, str(e)[:120]


def main(n_workers: int = 4) -> None:
    GLDAS_DIR.mkdir(parents=True, exist_ok=True)

    token = _load_env("NASA_EARTHDATA_TOKEN")
    user = _load_env("NASA_EARTHDATA_USER")
    password = _load_env("NASA_EARTHDATA_PASSWORD")
    if not token and (not user or not password):
        raise SystemExit(
            "Renseigne NASA_EARTHDATA_TOKEN (ou USER+PASSWORD) dans .env"
        )
    if token:
        logger.info("Authentification via Bearer token")
    else:
        logger.info("Authentification via Basic Auth (%s)", user)

    # Lire les URLs (ignorer les non-.nc4)
    urls = [
        line.strip()
        for line in URL_LIST.read_text().splitlines()
        if line.strip().endswith(".nc4")
    ]
    logger.info("%d fichiers listés dans %s", len(urls), URL_LIST.name)

    # Calculer ce qui reste à télécharger
    todo = []
    for url in urls:
        fname = url.split("/")[-1]
        dest = GLDAS_DIR / fname
        if _is_complete(dest):
            logger.debug("  skip %s", fname)
        else:
            todo.append((url, dest))

    already = len(urls) - len(todo)
    logger.info("%d déjà présents, %d à télécharger", already, len(todo))

    if not todo:
        logger.info("Tout est déjà téléchargé.")
        return

    session = _make_session(user, password, token)

    done = 0
    errors: list[str] = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_download_one, session, url, dest): dest.name
                   for url, dest in todo}
        for fut in as_completed(futures):
            fname, ok, msg = fut.result()
            done += 1
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            if ok:
                logger.info("  [%d/%d] ✓ %s — %s  (%.1f f/min)",
                            done, len(todo), fname, msg, rate * 60)
            else:
                logger.warning("  [%d/%d] ✗ %s — %s", done, len(todo), fname, msg)
                errors.append(f"{fname}: {msg}")

    logger.info("Terminé — %d/%d téléchargés", len(todo) - len(errors), len(todo))
    if errors:
        logger.warning("%d erreurs :", len(errors))
        for e in errors:
            logger.warning("  %s", e)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4,
                        help="Nombre de téléchargements parallèles (défaut: 4)")
    args = parser.parse_args()
    main(n_workers=args.workers)
