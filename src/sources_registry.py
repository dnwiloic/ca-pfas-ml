"""
Registre des 23 sources — Dong et al. 2024, Table S1 (Supporting Information).

Chaque entrée décrit une source pour `collect.py` :
  - kind ``url`` : fichier HTTP(S) direct
  - kind ``ckan`` : ressource data.ca.gov (package groundwater quality)
  - kind ``arcgis`` : export paginé FeatureServer → CSV
  - kind ``local`` : copie depuis un chemin du dépôt parent (ex. GAMA-PBP USGS)
  - kind ``manual`` : URL documentée ; téléchargement manuel si l’API échoue
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Kind = Literal["url", "ckan", "arcgis", "local", "manual"]


@dataclass(frozen=True)
class SourceSpec:
    id: str
    name: str
    category: Literal["gama", "contamination", "environment"]
    kind: Kind
    dest: str  # chemin relatif sous data/raw/
  # --- url / ckan ---
    url: str = ""
    resource_id: str = ""
  # --- arcgis ---
    arcgis_query_url: str = ""
  # --- local ---
    local_glob: str = ""
  # --- options ---
    notes: str = ""
    optional: bool = False
    large_mb: int = 0  # avertissement si > seuil --max-mb


PACKAGE_ID = "ground-water-water-quality-results"
PACKAGE_UUID = "81d1347c-891d-4f09-abc5-6eeb521b55d2"
CKAN_API = "https://data.ca.gov/api/3/action"

# 12 jeux GAMA (+ équivalent actuel pour DDW 2015-2019 → 2010-2019 sur data.ca.gov)
GAMA_CKAN: list[tuple[str, str, str]] = [
    ("gama_pfas_statewide", "c7cd50a2-2a68-44a1-8d40-4e767f363964", "gama/pfas.csv"),
    ("gama_all_wells", "805e6762-1b82-48d9-b68f-5d79cca06ace", "gama/gama_allwells.csv"),
    ("gama_dpr", "a0d400c0-fa18-4f2d-adbd-cbe2e552bca2", "gama/dpr.csv"),
    ("gama_ddw_2010_2019", "a34c79c3-698e-4047-ab4f-69efc456906c", "gama/ddw2010-2019.csv"),
    ("gama_ddw_2020_present", "d2e74ace-2cf4-4baf-aadd-406280bf1c1c", "gama/ddw2020-present.csv"),
    ("gama_wrd", "91657757-7c51-48f2-b154-9209b5c0616e", "gama/wrd.csv"),
    ("gama_ilrp", "c97e1fcf-8ad0-4913-9234-ba682ea73bc4", "gama/wbilrp.csv"),
    ("gama_usgs_nwis", "75ff1943-c37a-43c4-bf5f-b80532a88ed8", "gama/usgsnwis.csv"),
    ("gama_local_gw", "e7b53637-61a1-4c05-a835-63d8392423c3", "gama/localgw.csv"),
    ("gama_combined", "9e09ac40-b694-4ddc-acd2-2e7f7d10e231", "gama/gama.csv"),
    ("gama_dwr", "81b65d03-9b06-44cd-beae-d93f5b2718a0", "gama/dwr.csv"),
]


def build_sources() -> list[SourceSpec]:
    """Construit la liste complète des 23 sources."""
    out: list[SourceSpec] = []

    for sid, rid, dest in GAMA_CKAN:
        note = ""
        if sid == "gama_ddw_2010_2019":
            note = (
                "Article : DDW 2015-2019 (ressource be2d189b obsolète). "
                "Proxy : ddw2010-2019.csv sur data.ca.gov."
            )
        large = 500 if sid in ("gama_combined", "gama_dpr", "gama_ddw_2010_2019", "gama_ddw_2020_present") else 200
        out.append(
            SourceSpec(
                id=sid,
                name=dest.split("/")[-1],
                category="gama",
                kind="ckan",
                dest=dest,
                resource_id=rid,
                notes=note,
                large_mb=large,
            )
        )

    out.append(
        SourceSpec(
            id="gama_pbp_usgs",
            name="USGS GAMA-PBP PFAS 2019-2024",
            category="gama",
            kind="local",
            dest="gama/gama_pbp/",
            local_glob="../usgs2_GAMA_PBP/Table_*.txt",
            url="https://doi.org/10.5066/P1RQGZ68",
            notes="Copie locale des 5 tables USGS si présentes ; sinon télécharger via DOI.",
        )
    )

    # --- 4 sources contamination (EPA / GeoTracker / EWG) ---
    out.extend(
        [
            SourceSpec(
                id="geotracker_pfas_sites",
                name="GeoTracker PFAS — sites d'investigation",
                category="contamination",
                kind="arcgis",
                dest="contamination/geotracker_pfas_investigation_sites.csv",
                arcgis_query_url=(
                    "https://gispublic.waterboards.ca.gov/portalserver/rest/services/"
                    "Hosted/PFAS_Investigation_Sites_Combo/FeatureServer/0/query"
                ),
                notes="Proxy Table S1 GeoTracker PFAS Map (couche SWRCB publique).",
            ),
            SourceSpec(
                id="epa_pfas_industry_sectors",
                name="EPA PFAS Handling Industry Sectors",
                category="contamination",
                kind="url",
                dest="contamination/epa_pfas_industry_sectors.xlsx",
                url="https://echo.epa.gov/system/files/PFASHandlingIndustrySectors-Apr2023-Pub.xlsx",
            ),
            SourceSpec(
                id="epa_pfas_npdes_dmr",
                name="EPA PFAS NPDES DMR (national)",
                category="contamination",
                kind="manual",
                dest="contamination/epa_pfas_npdes_dmr/",
                url="https://echo.epa.gov/tools/data-downloads/national-pfas-datasets",
                notes=(
                    "Exporter l’onglet NPDES DMR depuis PFAS Analytic Tools ou "
                    "https://awsedap.epa.gov/public/extensions/PFAS_Metadata/PFAS_Metadata.html"
                ),
                optional=True,
                large_mb=500,
            ),
            SourceSpec(
                id="epa_pfas_tri_releases",
                name="EPA TRI PFAS on-site releases",
                category="contamination",
                kind="manual",
                dest="contamination/epa_pfas_tri_releases/",
                url="https://echo.epa.gov/tools/data-downloads/national-pfas-datasets",
                notes="Exporter l’onglet TRI depuis PFAS Analytic Tools (ECHO).",
                optional=True,
                large_mb=200,
            ),
            SourceSpec(
                id="epa_cdr_pfas",
                name="EPA Chemical Data Reporting (PFAS)",
                category="contamination",
                kind="manual",
                dest="contamination/epa_cdr_pfas/",
                url="https://www.epa.gov/chemical-data-reporting/access-cdr-data",
                notes="CDR 2012–2020 ; filtrer les enregistrements PFAS (CompTox lists).",
                optional=True,
                large_mb=1000,
            ),
        ]
    )

    # --- 7 jeux environnement / sol / météo ---
    out.extend(
        [
            SourceSpec(
                id="epa_air_emissions_combined",
                name="EPA Air Emissions (NEI/GHGRP/TRI/CAMD)",
                category="environment",
                kind="manual",
                dest="environment/epa_air_emissions/",
                url="https://echo.epa.gov/tools/data-downloads/air-emissions-download-summary",
                notes="Téléchargement national ZIP depuis ECHO (très volumineux).",
                optional=True,
                large_mb=2000,
            ),
            SourceSpec(
                id="ncss_soil_characterization",
                name="NCSS Soil Characterization Database",
                category="environment",
                kind="manual",
                dest="environment/ncss_soil/",
                url="https://ncsslabdatamart.sc.egov.usda.gov/querypage.aspx",
                notes="Requêtes par comté/pédons ; export manuel ou script USDA ultérieur.",
                optional=True,
            ),
            SourceSpec(
                id="noaa_ghcnd_daily",
                name="NOAA GHCND daily summaries (CA)",
                category="environment",
                kind="manual",
                dest="environment/noaa_ghcnd/",
                url="https://www.ncei.noaa.gov/maps/daily-summaries/",
                notes="Ou API CDO : https://www.ncei.noaa.gov/cdo-web/api/v2/ (token NOAA_CDO_TOKEN).",
                optional=True,
            ),
            SourceSpec(
                id="epa_aqs_outdoor_air",
                name="EPA Outdoor Air Quality (AQS daily)",
                category="environment",
                kind="manual",
                dest="environment/epa_aqs/",
                url="https://www.epa.gov/outdoor-air-quality-data/download-daily-data",
                notes="Compte AQS requis : https://aqs.epa.gov/aqsweb/documents/data_api.html",
                optional=True,
            ),
            SourceSpec(
                id="purpleair",
                name="PurpleAir API",
                category="environment",
                kind="manual",
                dest="environment/purpleair/",
                url="https://community.purpleair.com/t/making-api-calls-with-the-purpleair-api/180",
                notes="Clé API : PURPLEAIR_API_KEY dans .env",
                optional=True,
            ),
            SourceSpec(
                id="sgma_groundwater_basins",
                name="DWR SGMA Bulletin 118 groundwater basins",
                category="environment",
                kind="arcgis",
                dest="environment/sgma_basins.geojson",
                arcgis_query_url=(
                    "https://gis.water.ca.gov/arcgis/rest/services/Geoscientific/"
                    "i08_B118_CA_GroundwaterBasins/FeatureServer/0/query"
                ),
            ),
            SourceSpec(
                id="nasa_gldas_water_balance",
                name="NASA GLDAS / Water Balance App (SGMA hydrology)",
                category="environment",
                kind="manual",
                dest="environment/nasa_gldas_water_balance/",
                url="https://livingatlas.arcgis.com/waterbalance/",
                notes=(
                    "Runoff, précipitation, ET, humidité du sol ; "
                    "niveaux GW : https://sgma.water.ca.gov/webgis/?appid=SGMADataViewer#gwlevels"
                ),
            ),
        ]
    )
    return out


SOURCES: list[SourceSpec] = build_sources()
SOURCE_BY_ID = {s.id: s for s in SOURCES}
