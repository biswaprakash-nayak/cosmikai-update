import json
import math
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from pydantic import BaseModel


class StarDetailsResponse(BaseModel):
    star_name: str
    source: str
    ra: float | None = None
    dec: float | None = None
    gaia_id: str | None = None
    tic_id: str | None = None
    teff: float | None = None
    radius: float | None = None
    mass: float | None = None
    logg: float | None = None
    distance: float | None = None
    vmag: float | None = None
    tmag: float | None = None
    found: bool = False


class StarDetailsFetchError(RuntimeError):
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.status_code = status_code


MAST_INVOKE_URL = "https://mast.stsci.edu/api/v0/invoke"


def _mast_invoke(service: str, params: dict) -> dict:
    payload = {
        "service": service,
        "params": params,
        "format": "json",
        "pagesize": 25,
        "page": 1,
    }
    encoded_payload = urllib.parse.quote(json.dumps(payload))
    body = f"request={encoded_payload}".encode("utf-8")
    req = urllib.request.Request(
        MAST_INVOKE_URL,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "User-Agent": "CosmiKAI/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=12) as resp:
        raw = resp.read().decode("utf-8")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        if raw.lstrip().startswith("<"):
            try:
                root = ET.fromstring(raw)
                resolved_rows: list[dict] = []
                for coord in root.findall(".//resolvedCoordinate"):
                    resolved_rows.append(
                        {
                            "ra": coord.findtext("ra"),
                            "decl": coord.findtext("dec"),
                            "canonicalName": coord.findtext("canonicalName"),
                        }
                    )
                return {"resolvedCoordinate": resolved_rows}
            except ET.ParseError:
                pass

        snippet = raw[:180].replace("\n", " ")
        raise StarDetailsFetchError(
            f"MAST returned non-JSON response: {snippet}",
            status_code=502,
        ) from exc


def _to_float(value) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _pick_best_tic_row(rows: list[dict], ra: float, dec: float) -> dict | None:
    if not rows:
        return None

    def row_score(row: dict) -> tuple[float, float]:
        teff = _to_float(row.get("Teff"))
        radius = _to_float(row.get("rad"))
        mass = _to_float(row.get("mass"))
        logg = _to_float(row.get("logg"))
        completeness = float(sum(v is not None for v in (teff, radius, mass, logg)))

        row_ra = _to_float(row.get("ra"))
        row_dec = _to_float(row.get("dec"))
        if row_ra is None or row_dec is None:
            separation = 1e9
        else:
            # Small-angle approximation is enough for tiny cone radius.
            dra = (row_ra - ra) * max(0.1, abs(math.cos(math.radians(dec))))
            ddec = row_dec - dec
            separation = (dra * dra + ddec * ddec) ** 0.5

        # Prefer rows with richer stellar params, then nearest match.
        return (completeness, -separation)

    return max(rows, key=row_score)


def fetch_star_details_from_mast(star_name: str) -> StarDetailsResponse:
    try:
        name_lookup = _mast_invoke("Mast.Name.Lookup", {"input": star_name})
        name_rows = name_lookup.get("resolvedCoordinate") or name_lookup.get("data") or []
        if not name_rows:
            return StarDetailsResponse(star_name=star_name, source="MAST", found=False)

        coord = name_rows[0]
        ra = _to_float(coord.get("ra"))
        dec = _to_float(coord.get("decl"))

        if ra is None or dec is None:
            return StarDetailsResponse(star_name=star_name, source="MAST", found=False)

        tic_result = _mast_invoke(
            "Mast.Catalogs.Filtered.Tic.Position",
            {
                "ra": ra,
                "dec": dec,
                "radius": 0.02,
                "columns": "ID,GAIA,Teff,rad,mass,logg,d,Vmag,Tmag,ra,dec",
                "filters": [],
            },
        )
        tic_rows = tic_result.get("data") or []

        if not tic_rows:
            return StarDetailsResponse(star_name=star_name, source="MAST", ra=ra, dec=dec, found=True)

        row = _pick_best_tic_row(tic_rows, ra, dec) or tic_rows[0]
        return StarDetailsResponse(
            star_name=star_name,
            source="MAST:TIC",
            ra=_to_float(row.get("ra")) or ra,
            dec=_to_float(row.get("dec")) or dec,
            gaia_id=str(row.get("GAIA")) if row.get("GAIA") is not None else None,
            tic_id=str(row.get("ID")) if row.get("ID") is not None else None,
            teff=_to_float(row.get("Teff")),
            radius=_to_float(row.get("rad")),
            mass=_to_float(row.get("mass")),
            logg=_to_float(row.get("logg")),
            distance=_to_float(row.get("d")),
            vmag=_to_float(row.get("Vmag")),
            tmag=_to_float(row.get("Tmag")),
            found=True,
        )
    except urllib.error.HTTPError as exc:
        raise StarDetailsFetchError(f"MAST HTTP error: {exc.code}", status_code=502)
    except urllib.error.URLError as exc:
        raise StarDetailsFetchError(f"MAST connection error: {exc.reason}", status_code=502)
    except Exception as exc:
        raise StarDetailsFetchError(f"Unable to fetch MAST star details: {exc}", status_code=500)
