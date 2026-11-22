"""Weather and environment loaders, site-date matching, covariate residualization,
and mechanistic thermal calculations for case study lpd (Protocol Amendment 1).
Run via analysis.py under python3 -I.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import math
from pathlib import Path
import tarfile
import numpy as np

from helpers import (
    invert_bias_corrected_r1,
    invert_projected_bias_corrected_r1,
    ols_detrend,
    orthonormal_basis_from_design,
    projected_white_noise_r1_null_mean,
    thermal_loss_d2_one_exposed,
    white_noise_r1_null_mean,
)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r_earth = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * r_earth * math.asin(min(1.0, math.sqrt(a)))


class WeatherArchive:
    """In-memory index of downloaded open weather and sea-surface datasets under data/raw/lpd/weather/."""

    def __init__(self, weather_dir: Path, needed_land_sites: list[tuple[float, float]]) -> None:
        self.weather_dir = weather_dir
        self.daymet: dict[str, dict] = {}
        self.terraclimate: dict[tuple[int, int], dict] = {}
        self.ersst_cache: dict[int, np.ndarray] = {}
        self.ghcn_stations: list[tuple[str, float, float, float]] = []
        self.ghcn_data: dict[str, dict[tuple[int, int], float]] = {}

        self._load_daymet()
        self._load_terraclimate()
        self._load_ghcnm(needed_land_sites)

    def _load_daymet(self) -> None:
        dm_dir = self.weather_dir / "daymet"
        if not dm_dir.exists():
            return
        for fp in sorted(dm_dir.glob("*.csv")):
            key = fp.stem  # e.g. VD_27224
            lines = fp.read_text(encoding="utf-8", errors="replace").splitlines()
            pix_lat, pix_lon = float("nan"), float("nan")
            header_idx = None
            for idx, ln in enumerate(lines[:15]):
                if ln.startswith("Latitude:"):
                    toks = ln.replace("Latitude:", " ").replace("Longitude:", " ").split()
                    if len(toks) >= 2:
                        try:
                            pix_lat, pix_lon = float(toks[0]), float(toks[1])
                        except ValueError:
                            pass
                if ln.startswith("year,yday"):
                    header_idx = idx
                    break
            if header_idx is None:
                continue
            by_date: dict[dt.date, tuple[float, float]] = {}
            rd = csv.DictReader(lines[header_idx:])
            for r in rd:
                try:
                    yr = int(r["year"])
                    yd = int(r["yday"])
                    tmax = float(r["tmax (deg c)"])
                    tmin = float(r["tmin (deg c)"])
                    prcp = float(r["prcp (mm/day)"])
                except (ValueError, KeyError, TypeError):
                    continue
                dd = dt.date(yr, 1, 1) + dt.timedelta(days=yd - 1)
                by_date[dd] = (0.5 * (tmax + tmin), prcp)
            self.daymet[key] = {
                "cell_lat": pix_lat,
                "cell_lon": pix_lon,
                "by_date": by_date,
            }

    def _load_terraclimate(self) -> None:
        tc_dir = self.weather_dir / "terraclimate"
        if not tc_dir.exists():
            return
        for fp in sorted(tc_dir.glob("tc_*.json")):
            # Filename encodes the requested reference (ilat, ilon): tc_{ilat}_{ilon}_{y0}_{y1}.json
            parts = fp.stem.split("_")
            req_ilat, req_ilon = int(parts[1]), int(parts[2])
            d = json.loads(fp.read_text(encoding="utf-8"))
            if not d.get("valid_land"):
                continue
            t0 = int(d["t0_idx"])
            tmax = d["tmax_c"]
            tmin = d["tmin_c"]
            ppt = d["ppt_mm"]
            ym_map: dict[tuple[int, int], tuple[float, float | None]] = {}
            for k in range(len(tmax)):
                if tmax[k] is None or tmin[k] is None:
                    continue
                abs_m = t0 + k
                yr = 1950 + abs_m // 12
                mo = (abs_m % 12) + 1
                ym_map[(yr, mo)] = (0.5 * (float(tmax[k]) + float(tmin[k])), float(ppt[k]) if ppt[k] is not None else None)
            if (req_ilat, req_ilon) in self.terraclimate:
                self.terraclimate[(req_ilat, req_ilon)]["ym"].update(ym_map)
            else:
                self.terraclimate[(req_ilat, req_ilon)] = {
                    "cell_lat": float(d["cell_lat"]),
                    "cell_lon": float(d["cell_lon"]),
                    "ym": ym_map,
                }

    def _load_ghcnm(self, needed_land_sites: list[tuple[float, float]]) -> None:
        tar_path = self.weather_dir / "ghcnm" / "ghcnm.tavg.latest.qcf.tar.gz"
        if not tar_path.exists() or not needed_land_sites:
            return
        with tarfile.open(tar_path, "r:gz") as tf:
            inv_member = next((m for m in tf.getmembers() if m.name.endswith(".inv")), None)
            dat_member = next((m for m in tf.getmembers() if m.name.endswith(".dat")), None)
            if inv_member is None or dat_member is None:
                return
            all_stns: list[tuple[str, float, float, float]] = []
            inv_f = tf.extractfile(inv_member)
            assert inv_f is not None
            for raw_ln in inv_f:
                ln = raw_ln.decode("latin-1")
                if len(ln) < 37:
                    continue
                sid = ln[0:11].strip()
                try:
                    slat = float(ln[12:20])
                    slon = float(ln[21:30])
                    selev = float(ln[31:37])
                except ValueError:
                    continue
                all_stns.append((sid, slat, slon, selev))

            # Keep top-8 nearest candidate stations within 250 km of each distinct land study site
            stn_lats = np.array([s[1] for s in all_stns], dtype=float)
            stn_lons = np.array([s[2] for s in all_stns], dtype=float)
            wanted_ids: set[str] = set()
            kept_map: dict[str, tuple[str, float, float, float]] = {}
            for lat0, lon0 in needed_land_sites:
                # Fast bounding-box prefilter before exact Haversine
                dlat = np.abs(stn_lats - lat0)
                cand_idx = np.flatnonzero(dlat <= 2.5)
                if len(cand_idx) == 0:
                    continue
                dists = [
                    (haversine_km(lat0, lon0, float(stn_lats[i]), float(stn_lons[i])), int(i))
                    for i in cand_idx
                ]
                dists.sort(key=lambda x: x[0])
                for dist_km, idx_s in dists[:8]:
                    if dist_km <= 250.0:
                        st_tuple = all_stns[idx_s]
                        wanted_ids.add(st_tuple[0])
                        kept_map[st_tuple[0]] = st_tuple

            self.ghcn_stations = list(kept_map.values())
            dat_f = tf.extractfile(dat_member)
            assert dat_f is not None
            for raw_ln in dat_f:
                sid = raw_ln[0:11].decode("latin-1").strip()
                if sid not in wanted_ids:
                    continue
                ln = raw_ln.decode("latin-1")
                try:
                    yr = int(ln[11:15])
                except ValueError:
                    continue
                sdict = self.ghcn_data.setdefault(sid, {})
                for m_idx in range(12):
                    pos = 19 + 8 * m_idx
                    if pos + 7 > len(ln):
                        break
                    val_s = ln[pos : pos + 5].strip()
                    qflag = ln[pos + 6]
                    if not val_s or val_s == "-9999" or qflag != " ":
                        continue
                    try:
                        sdict[(yr, m_idx + 1)] = int(val_s) * 0.01
                    except ValueError:
                        pass

    def _get_ersst_year(self, yr: int) -> np.ndarray | None:
        if yr in self.ersst_cache:
            return self.ersst_cache[yr]
        fp = self.weather_dir / "ersst_v5" / f"ersst.v5.{yr}.asc"
        if not fp.exists():
            return None
        arr = np.fromstring(fp.read_text(encoding="ascii", errors="ignore"), dtype=np.int32, sep=" ")
        if arr.size != 12 * 180 * 89:
            return None
        cube = arr.reshape((12, 180, 89))
        self.ersst_cache[yr] = cube
        return cube

    def match_ersst_cell(self, lat: float, lon: float, ref_year: int) -> tuple[int, int, float, float, float] | None:
        """Find nearest valid non-land ERSST v5 2x2-deg cell within 250 km."""
        cube = self._get_ersst_year(ref_year)
        if cube is None and self.ersst_cache:
            cube = next(iter(self.ersst_cache.values()))
        if cube is None:
            # Try loading 2000 as land-mask reference
            cube = self._get_ersst_year(2000)
        if cube is None:
            return None
        lon360 = lon % 360.0
        i0 = int(round(lon360 / 2.0)) % 180
        j0 = max(0, min(88, int(round((lat + 88.0) / 2.0))))
        best = None
        best_d = 1e9
        for di in range(-2, 3):
            for dj in range(-2, 3):
                ii = (i0 + di) % 180
                jj = j0 + dj
                if not (0 <= jj <= 88):
                    continue
                if int(cube[0, ii, jj]) <= -9990:
                    continue
                clat = -88.0 + 2.0 * jj
                clon_raw = 2.0 * ii
                clon = clon_raw - 360.0 if clon_raw > 180.0 else clon_raw
                dkm = haversine_km(lat, lon, clat, clon)
                if dkm <= 250.0 and dkm < best_d:
                    best_d = dkm
                    best = (ii, jj, clat, clon, dkm)
        return best

    def extract_series_weather(self, s: dict, pos_mask: np.ndarray) -> dict:
        """Attach weather time series on the exact observed positive host steps s['steps'][pos_mask]."""
        if s.get("environment_type") == "laboratory":
            return {"matched": False, "source": "laboratory_regime", "reason": "indoor_laboratory"}

        lat = float(s.get("latitude", float("nan")))
        lon = float(s.get("longitude", float("nan")))
        if math.isnan(lat) or math.isnan(lon) or not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            return {"matched": False, "source": "unmatched", "reason": "missing_coordinates"}

        t_steps = s["steps"][pos_mask]
        yrs = s["step_years"][pos_mask]
        mos = s["step_months"][pos_mask]
        valid_yr = (yrs >= 1700) & (yrs <= 2026)
        if int(np.sum(valid_yr)) < 12:
            return {"matched": False, "source": "unmatched", "reason": "missing_calendar_dates"}

        iv = s["interval_class"]
        realm = s.get("realm_detail", "Terrestrial")
        n_pts = len(t_steps)

        # --- Route 1: North American sub-annual VecDyn -> Daymet 1-km daily ---
        if s["database"] == "VecDyn":
            did_key = s["study_id"].replace("VD_SET_", "VD_")
            if did_key in self.daymet:
                dm = self.daymet[did_key]
                by_d = dm["by_date"]
                temps = np.full(n_pts, np.nan, dtype=float)
                prcps = np.full(n_pts, np.nan, dtype=float)
                bdays = int(s.get("bin_days", 0))
                dates_iso = s.get("step_dates_iso")
                if bdays in (7, 14) and dates_iso is not None:
                    pos_indices = np.flatnonzero(pos_mask)
                    for k_i, orig_i in enumerate(pos_indices):
                        d0 = dt.date.fromisoformat(dates_iso[int(orig_i)])
                        vals_t, vals_p = [], []
                        for off in range(bdays):
                            rec = by_d.get(d0 + dt.timedelta(days=off))
                            if rec is not None:
                                vals_t.append(rec[0])
                                vals_p.append(rec[1])
                        if vals_t:
                            temps[k_i] = float(np.mean(vals_t))
                            prcps[k_i] = float(np.sum(vals_p))
                elif iv == "subannual_monthly":
                    for k_i in range(n_pts):
                        yr_k, mo_k = int(yrs[k_i]), int(mos[k_i])
                        vals_t, vals_p = [], []
                        for dom in range(1, 32):
                            try:
                                dd = dt.date(yr_k, mo_k, dom)
                            except ValueError:
                                break
                            rec = by_d.get(dd)
                            if rec is not None:
                                vals_t.append(rec[0])
                                vals_p.append(rec[1])
                        if vals_t:
                            temps[k_i] = float(np.mean(vals_t))
                            prcps[k_i] = float(np.sum(vals_p))
                if int(np.sum(~np.isnan(temps))) >= 12:
                    dkm = haversine_km(lat, lon, dm["cell_lat"], dm["cell_lon"])
                    return self._finalize_weather_metrics(
                        s, pos_mask, temps, prcps,
                        source="Daymet_V4R1_1km_daily",
                        resolution="daily 1 km aggregated to sampling bin",
                        mlat=dm["cell_lat"], mlon=dm["cell_lon"], dist_km=dkm,
                    )

        # --- Route 2: Marine / Estuarine -> NOAA ERSST v5 2x2-deg monthly SST ---
        if realm == "Marine":
            ref_yr = int(np.median(yrs[valid_yr]))
            cell = self.match_ersst_cell(lat, lon, ref_yr)
            if cell is not None:
                ii, jj, clat, clon, dkm = cell
                temps = np.full(n_pts, np.nan, dtype=float)
                for k_i in range(n_pts):
                    yr_k, mo_k = int(yrs[k_i]), int(mos[k_i])
                    cube = self._get_ersst_year(yr_k)
                    if cube is None:
                        continue
                    if iv == "annual" or mo_k == 0:
                        row_m = cube[:, ii, jj]
                        ok_m = row_m > -9990
                        if int(np.sum(ok_m)) >= 6:
                            temps[k_i] = float(np.mean(row_m[ok_m]) * 0.01)
                    elif iv == "subannual_seasonal":
                        q_start = max(0, min(9, ((mo_k - 1) // 3) * 3))
                        row_m = cube[q_start : q_start + 3, ii, jj]
                        ok_m = row_m > -9990
                        if int(np.sum(ok_m)) >= 1:
                            temps[k_i] = float(np.mean(row_m[ok_m]) * 0.01)
                    else:
                        v_raw = int(cube[max(0, min(11, mo_k - 1)), ii, jj])
                        if v_raw > -9990:
                            temps[k_i] = v_raw * 0.01
                if int(np.sum(~np.isnan(temps))) >= 12:
                    return self._finalize_weather_metrics(
                        s, pos_mask, temps, None,
                        source="NOAA_ERSST_v5_2deg_SST",
                        resolution="monthly 2x2 deg global SST",
                        mlat=clat, mlon=clon, dist_km=dkm,
                    )

        # --- Route 3: Terrestrial & Freshwater 1950-2025 -> TerraClimate 1/24-deg (~4 km) monthly ---
        ilat = max(1, min(4318, int(round((90.0 - lat) * 24.0 - 0.5))))
        ilon = max(1, min(8638, int(round((lon + 180.0) * 24.0 - 0.5))))
        tc = self.terraclimate.get((ilat, ilon))
        if tc is None:
            # Check immediate 1-cell neighbor key if rounded differently
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    tc = self.terraclimate.get((ilat + di, ilon + dj))
                    if tc is not None:
                        break
                if tc is not None:
                    break
        if tc is not None:
            ym = tc["ym"]
            temps = np.full(n_pts, np.nan, dtype=float)
            prcps = np.full(n_pts, np.nan, dtype=float)
            for k_i in range(n_pts):
                yr_k, mo_k = int(yrs[k_i]), int(mos[k_i])
                if iv == "annual" or mo_k == 0:
                    vt, vp = [], []
                    for mm in range(1, 13):
                        rec = ym.get((yr_k, mm))
                        if rec is not None:
                            vt.append(rec[0])
                            if rec[1] is not None:
                                vp.append(rec[1])
                    if len(vt) >= 6:
                        temps[k_i] = float(np.mean(vt))
                        if len(vp) >= 6:
                            prcps[k_i] = float(np.sum(vp))
                elif iv == "subannual_seasonal":
                    q0 = ((max(1, mo_k) - 1) // 3) * 3 + 1
                    vt, vp = [], []
                    for mm in (q0, q0 + 1, q0 + 2):
                        rec = ym.get((yr_k, mm))
                        if rec is not None:
                            vt.append(rec[0])
                            if rec[1] is not None:
                                vp.append(rec[1])
                    if vt:
                        temps[k_i] = float(np.mean(vt))
                        if vp:
                            prcps[k_i] = float(np.sum(vp))
                else:
                    rec = ym.get((yr_k, max(1, min(12, mo_k))))
                    if rec is not None:
                        temps[k_i] = rec[0]
                        if rec[1] is not None:
                            prcps[k_i] = rec[1]
            if int(np.sum(~np.isnan(temps))) >= 12:
                dkm = haversine_km(lat, lon, tc["cell_lat"], tc["cell_lon"])
                out = self._finalize_weather_metrics(
                    s, pos_mask, temps, prcps,
                    source="TerraClimate_4km_monthly",
                    resolution="monthly 1/24 deg (~4 km)",
                    mlat=tc["cell_lat"], mlon=tc["cell_lon"], dist_km=dkm,
                )
                # Attach independent GHCNm station cross-check if a station within 100 km overlaps >= 12 steps
                ghcn_chk = self._match_ghcn_series(lat, lon, yrs, mos, iv, max_km=100.0)
                if ghcn_chk is not None:
                    t_g = ghcn_chk["temps"]
                    both = (~np.isnan(temps)) & (~np.isnan(t_g))
                    if int(np.sum(both)) >= 12 and float(np.std(temps[both])) > 1e-6 and float(np.std(t_g[both])) > 1e-6:
                        out["ghcn_crosscheck_station"] = ghcn_chk["station_id"]
                        out["ghcn_crosscheck_dist_km"] = round(ghcn_chk["dist_km"], 2)
                        out["ghcn_crosscheck_r"] = float(np.corrcoef(temps[both], t_g[both])[0, 1])
                return out

        # --- Route 4: Historical pre-1950 & coastal/island fallback -> NOAA GHCN-Monthly v4 station ---
        ghcn = self._match_ghcn_series(lat, lon, yrs, mos, iv, max_km=250.0)
        if ghcn is not None:
            return self._finalize_weather_metrics(
                s, pos_mask, ghcn["temps"], None,
                source=f"NOAA_GHCNm_v4_{ghcn['station_id']}",
                resolution="monthly meteorological station",
                mlat=ghcn["slat"], mlon=ghcn["slon"], dist_km=ghcn["dist_km"],
            )

        return {"matched": False, "source": "unmatched", "reason": "no_overlapping_station_or_grid_within_250km"}

    def _match_ghcn_series(
        self, lat: float, lon: float, yrs: np.ndarray, mos: np.ndarray, iv: str, max_km: float = 250.0
    ) -> dict | None:
        cands = []
        for sid, slat, slon, _ in self.ghcn_stations:
            if abs(slat - lat) > 2.5:
                continue
            dkm = haversine_km(lat, lon, slat, slon)
            if dkm <= max_km:
                cands.append((dkm, sid, slat, slon))
        cands.sort(key=lambda x: x[0])
        n_pts = len(yrs)
        for dkm, sid, slat, slon in cands[:8]:
            sdict = self.ghcn_data.get(sid, {})
            if not sdict:
                continue
            temps = np.full(n_pts, np.nan, dtype=float)
            for k_i in range(n_pts):
                yr_k, mo_k = int(yrs[k_i]), int(mos[k_i])
                if iv == "annual" or mo_k == 0:
                    vt = [sdict[(yr_k, mm)] for mm in range(1, 13) if (yr_k, mm) in sdict]
                    if len(vt) >= 8:
                        temps[k_i] = float(np.mean(vt))
                elif iv == "subannual_seasonal":
                    q0 = ((max(1, mo_k) - 1) // 3) * 3 + 1
                    vt = [sdict[(yr_k, mm)] for mm in (q0, q0 + 1, q0 + 2) if (yr_k, mm) in sdict]
                    if vt:
                        temps[k_i] = float(np.mean(vt))
                else:
                    v = sdict.get((yr_k, max(1, min(12, mo_k))))
                    if v is not None:
                        temps[k_i] = v
            if int(np.sum(~np.isnan(temps))) >= 12:
                return {"station_id": sid, "slat": slat, "slon": slon, "dist_km": dkm, "temps": temps}
        return None

    def _finalize_weather_metrics(
        self,
        s: dict,
        pos_mask: np.ndarray,
        temps: np.ndarray,
        prcps: np.ndarray | None,
        source: str,
        resolution: str,
        mlat: float,
        mlon: float,
        dist_km: float,
    ) -> dict:
        t_full = s["steps"][pos_mask].astype(float)
        y_full = np.log(s["values"][pos_mask])
        yrs_full = s["step_years"][pos_mask]

        ok = ~np.isnan(temps)
        t_w = t_full[ok]
        temp_w = temps[ok]
        y_w = y_full[ok]
        n_w = len(t_w)

        # Temperature time-series estimands on t_w
        e_temp, _, _ = ols_detrend(t_w, temp_w)
        sig2_t = float((e_temp @ e_temp) / max(n_w - 2, 1))
        sig_t = math.sqrt(max(sig2_t, 0.0))
        c0_t = float((e_temp @ e_temp) / n_w)
        adj_w = np.flatnonzero(np.diff(t_w) == 1.0)
        if len(adj_w) >= 8 and c0_t > 1e-12:
            r1_t = float((e_temp[adj_w] @ e_temp[adj_w + 1]) / (len(adj_w) * c0_t))
            mu0_t = white_noise_r1_null_mean(t_w)
            delta_r1_t = r1_t - mu0_t
            r1_bc_t = invert_bias_corrected_r1(t_w, r1_t)
        else:
            r1_t = float("nan")
            delta_r1_t = float("nan")
            r1_bc_t = float("nan")

        # Detrended host residuals on the matched weather steps
        e_host, _, _ = ols_detrend(t_w, y_w)
        if sig_t > 1e-9 and float(np.std(e_host)) > 1e-9:
            corr_ht = float(np.corrcoef(e_host, e_temp)[0, 1])
        else:
            corr_ht = 0.0
        if len(adj_w) >= 8 and sig_t > 1e-9 and float(np.std(e_host[adj_w + 1])) > 1e-9:
            corr_ht_lag1 = float(np.corrcoef(e_host[adj_w + 1], e_temp[adj_w])[0, 1])
        else:
            corr_ht_lag1 = float("nan")

        # Weather-purged host residuals u_k (Sensitivity S10 / W1b):
        # Regress y_w on [1, t_w, temp_w, prcp_log (if complete), lag1_temp (where available)]
        cols = [np.ones(n_w, dtype=float), t_w - t_w.mean(), temp_w - temp_w.mean()]
        # Construct lag-1 temperature predictor (imputed to mean 0 at non-adjacent starts so all n_w steps stay on grid)
        lag1_t = np.zeros(n_w, dtype=float)
        if len(adj_w) >= 8:
            lag1_t[adj_w + 1] = e_temp[adj_w]
            cols.append(lag1_t)
        mean_prcp = float("nan")
        if prcps is not None:
            p_ok = prcps[ok]
            if int(np.sum(~np.isnan(p_ok))) == n_w:
                mean_prcp = float(np.mean(p_ok))
                lp = np.log(np.maximum(p_ok, 0.0) + 1.0)
                if float(np.std(lp)) > 1e-8:
                    cols.append(lp - lp.mean())
        X = np.column_stack(cols)
        # Only fit weather regression if n_w > X.shape[1] + 5
        sigma_res, r1_res, delta_r1_res, delta_r1_res_2col, r1_bc_res, r2_w = (
            float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
        )
        if n_w >= max(15, X.shape[1] + 6) and len(adj_w) >= 10:
            beta_w, _, _, _ = np.linalg.lstsq(X, y_w, rcond=None)
            u_res = y_w - X @ beta_w
            ss_res = float(u_res @ u_res)
            ss_det = float(e_host @ e_host)
            r2_w = max(0.0, 1.0 - ss_res / max(ss_det, 1e-15))
            Q_x = orthonormal_basis_from_design(X)
            df_res = max(n_w - Q_x.shape[1], 1)
            sigma_res = math.sqrt(ss_res / df_res)
            c0_u = ss_res / n_w
            if c0_u > 1e-15:
                r1_res = float((u_res[adj_w] @ u_res[adj_w + 1]) / (len(adj_w) * c0_u))
                # Exact projection-matrix white-noise null & AR(1) bias correction for design X
                mu0_u = projected_white_noise_r1_null_mean(t_w, Q_x)
                delta_r1_res = r1_res - mu0_u
                delta_r1_res_2col = r1_res - white_noise_r1_null_mean(t_w)
                r1_bc_res = invert_projected_bias_corrected_r1(t_w, Q_x, r1_res)

        # Mechanistic thermal losses D_{2,T} (Statement C3c, mechanism-temperature-incubation.md §3.2)
        phi_t_use = float(np.clip(r1_bc_t, -0.95, 0.98)) if not math.isnan(r1_bc_t) else 0.0
        d2t_same_mod = thermal_loss_d2_one_exposed(1.0, 7.5, sig_t, phi_t_use, 0.05, 0.08)
        d2t_opp_mod = thermal_loss_d2_one_exposed(1.0, 7.5, sig_t, phi_t_use, -0.05, 0.08)
        d2t_same_short = thermal_loss_d2_one_exposed(4.0, 9.2, sig_t, (phi_t_use ** 0.25 if phi_t_use > 0 else 0.0), 0.05, 0.08)
        d2t_opp_short = thermal_loss_d2_one_exposed(4.0, 9.2, sig_t, (phi_t_use ** 0.25 if phi_t_use > 0 else 0.0), -0.05, 0.08)

        return {
            "matched": True,
            "source": source,
            "resolution": resolution,
            "matched_lat": round(mlat, 5),
            "matched_lon": round(mlon, 5),
            "distance_km": round(dist_km, 2),
            "matched_start_year": int(np.min(yrs_full[ok])),
            "matched_end_year": int(np.max(yrs_full[ok])),
            "n_matched_steps": int(n_w),
            "mean_temp_c": float(np.mean(temp_w)),
            "sigma_temp_c": sig_t,
            "r1_temp": r1_t,
            "delta_r1_temp": delta_r1_t,
            "r1_bc_temp": r1_bc_t,
            "mean_prcp_mm": mean_prcp,
            "corr_host_temp": corr_ht,
            "corr_host_lag1_temp": corr_ht_lag1,
            "r2_weather_explained": r2_w,
            "sigma_weather_resid": sigma_res,
            "r1_weather_resid": r1_res,
            "delta_r1_weather_resid": delta_r1_res,
            "delta_r1_weather_resid_2col": delta_r1_res_2col,
            "r1_bc_weather_resid": r1_bc_res,
            "d2_thermal_same_mod": d2t_same_mod,
            "d2_thermal_opp_mod": d2t_opp_mod,
            "d2_thermal_same_short_fine": d2t_same_short,
            "d2_thermal_opp_short_fine": d2t_opp_short,
        }
