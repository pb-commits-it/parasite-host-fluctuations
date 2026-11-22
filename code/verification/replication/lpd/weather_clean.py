"""Clean-room weather archive loader and site-date matcher for independent
replication of case study `lpd` Weather & Environment Amendment (`W1`).

Reads raw weather files directly from `data/raw/lpd/weather/` (`daymet/`,
`terraclimate/`, `ersst_v5/`, `ghcnm/`) without importing original study code.
"""
from __future__ import annotations

import calendar
import csv
import datetime as dt
import json
import math
from pathlib import Path
import tarfile
import numpy as np

import core_math as cm


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    earth_r = 6371.0088
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    h = math.sin(0.5 * dp) ** 2 + math.cos(p1) * math.cos(p2) * (math.sin(0.5 * dl) ** 2)
    return 2.0 * earth_r * math.asin(min(1.0, math.sqrt(h)))


class CleanWeatherStore:
    """Loads cached open weather datasets from `data/raw/lpd/weather/`.

    `cutoff_epoch`: if provided, ignores files added after that UTC timestamp
    so we can audit both the exact pre-Amendment-3 archive (when `report.md`
    Section 8 was generated) and the current on-disk archive after `r27`'s partial fetch.
    """

    def __init__(
        self,
        weather_dir: Path,
        land_coords: list[tuple[float, float]],
        cutoff_epoch: float | None = None,
        ghcn_max_Rank: int = 8,
    ) -> None:
        self.weather_dir = weather_dir
        self.cutoff_epoch = cutoff_epoch
        self.ghcn_max_rank = ghcn_max_Rank
        self.daymet: dict[str, dict] = {}
        self.terraclimate: dict[tuple[int, int], dict] = {}
        self.ersst_years: dict[int, np.ndarray] = {}
        self.ghcn_stations: list[tuple[str, float, float]] = []
        self.ghcn_monthly: dict[str, dict[tuple[int, int], float]] = {}

        self._load_daymet()
        self._load_terraclimate()
        self._load_ghcnm(land_coords)

    def _keep_file(self, fp: Path) -> bool:
        if self.cutoff_epoch is None:
            return True
        return fp.stat().st_mtime <= self.cutoff_epoch

    def _load_daymet(self) -> None:
        d_dir = self.weather_dir / "daymet"
        if not d_dir.exists():
            return
        for fp in sorted(d_dir.glob("*.csv")):
            if not self._keep_file(fp):
                continue
            lines = fp.read_text(encoding="utf-8", errors="replace").splitlines()
            clat, clon = float("nan"), float("nan")
            hdr_i = None
            for i, ln in enumerate(lines[:15]):
                if ln.startswith("Latitude:"):
                    parts = ln.replace("Latitude:", " ").replace("Longitude:", " ").split()
                    if len(parts) >= 2:
                        try:
                            clat, clon = float(parts[0]), float(parts[1])
                        except ValueError:
                            pass
                if ln.startswith("year,yday"):
                    hdr_i = i
                    break
            if hdr_i is None:
                continue
            daily: dict[dt.date, tuple[float, float]] = {}
            for r in csv.DictReader(lines[hdr_i:]):
                try:
                    yr = int(r["year"])
                    yd = int(r["yday"])
                    tmax = float(r["tmax (deg c)"])
                    tmin = float(r["tmin (deg c)"])
                    prcp = float(r["prcp (mm/day)"])
                except (ValueError, KeyError, TypeError):
                    continue
                d_obs = dt.date(yr, 1, 1) + dt.timedelta(days=yd - 1)
                daily[d_obs] = (0.5 * (tmax + tmin), prcp)
            self.daymet[fp.stem] = {"cell_lat": clat, "cell_lon": clon, "by_date": daily}

    def _load_terraclimate(self) -> None:
        tc_dir = self.weather_dir / "terraclimate"
        if not tc_dir.exists():
            return
        for fp in sorted(tc_dir.glob("tc_*.json")):
            if not self._keep_file(fp):
                continue
            toks = fp.stem.split("_")
            ii, jj = int(toks[1]), int(toks[2])
            d = json.loads(fp.read_text(encoding="utf-8"))
            if not d.get("valid_land"):
                continue
            t0 = int(d["t0_idx"])
            tmax = d["tmax_c"]
            tmin = d["tmin_c"]
            ppt = d["ppt_mm"]
            ym: dict[tuple[int, int], tuple[float, float | None]] = {}
            for k in range(len(tmax)):
                if tmax[k] is None or tmin[k] is None:
                    continue
                m_abs = t0 + k
                yr = 1950 + (m_abs // 12)
                mo = (m_abs % 12) + 1
                ym[(yr, mo)] = (0.5 * (float(tmax[k]) + float(tmin[k])), float(ppt[k]) if ppt[k] is not None else None)
            if (ii, jj) in self.terraclimate:
                self.terraclimate[(ii, jj)]["ym"].update(ym)
            else:
                self.terraclimate[(ii, jj)] = {
                    "cell_lat": float(d["cell_lat"]),
                    "cell_lon": float(d["cell_lon"]),
                    "ym": ym,
                }

    def _load_ghcnm(self, land_coords: list[tuple[float, float]]) -> None:
        tar_fp = self.weather_dir / "ghcnm" / "ghcnm.tavg.latest.qcf.tar.gz"
        if not tar_fp.exists() or not land_coords:
            return
        with tarfile.open(tar_fp, "r:gz") as tf:
            inv_m = next((m for m in tf.getmembers() if m.name.endswith(".inv")), None)
            dat_m = next((m for m in tf.getmembers() if m.name.endswith(".dat")), None)
            if inv_m is None or dat_m is None:
                return
            stns: list[tuple[str, float, float]] = []
            inv_f = tf.extractfile(inv_m)
            assert inv_f is not None
            for raw in inv_f:
                ln = raw.decode("latin-1")
                if len(ln) < 31:
                    continue
                sid = ln[0:11].strip()
                try:
                    slat = float(ln[12:20])
                    slon = float(ln[21:30])
                except ValueError:
                    continue
                stns.append((sid, slat, slon))

            s_lats = np.array([x[1] for x in stns], dtype=float)
            s_lons = np.array([x[2] for x in stns], dtype=float)
            keep_ids: set[str] = set()
            keep_stns: dict[str, tuple[str, float, float]] = {}
            for lat0, lon0 in land_coords:
                idx = np.flatnonzero(np.abs(s_lats - lat0) <= 2.5)
                dists = [(haversine_km(lat0, lon0, float(s_lats[i]), float(s_lons[i])), int(i)) for i in idx]
                dists.sort(key=lambda x: x[0])
                cand_slice = dists[: self.ghcn_max_rank] if self.ghcn_max_rank > 0 else dists
                for dkm, ii in cand_slice:
                    if dkm <= 250.0:
                        st = stns[ii]
                        keep_ids.add(st[0])
                        keep_stns[st[0]] = st

            self.ghcn_stations = list(keep_stns.values())
            dat_f = tf.extractfile(dat_m)
            assert dat_f is not None
            for raw in dat_f:
                sid = raw[0:11].decode("latin-1").strip()
                if sid not in keep_ids:
                    continue
                ln = raw.decode("latin-1")
                try:
                    yr = int(ln[11:15])
                except ValueError:
                    continue
                ym = self.ghcn_monthly.setdefault(sid, {})
                for m_i in range(12):
                    p0 = 19 + 8 * m_i
                    if p0 + 7 > len(ln):
                        break
                    vs = ln[p0 : p0 + 5].strip()
                    qf = ln[p0 + 6]
                    if not vs or vs == "-9999" or qf != " ":
                        continue
                    try:
                        ym[(yr, m_i + 1)] = int(vs) * 0.01
                    except ValueError:
                        pass

    def ersst_cube(self, yr: int) -> np.ndarray | None:
        if yr in self.ersst_years:
            return self.ersst_years[yr]
        fp = self.weather_dir / "ersst_v5" / f"ersst.v5.{yr}.asc"
        if not fp.exists() or not self._keep_file(fp):
            return None
        arr = np.fromstring(fp.read_text(encoding="ascii", errors="ignore"), dtype=np.int32, sep=" ")
        if arr.size != 12 * 180 * 89:
            return None
        cb = arr.reshape((12, 180, 89))
        self.ersst_years[yr] = cb
        return cb

    def nearest_ersst(self, lat: float, lon: float, ref_yr: int) -> tuple[int, int, float, float, float] | None:
        cb = self.ersst_cube(ref_yr)
        if cb is None:
            cb = self.ersst_cube(2000)
        if cb is None:
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
                if 0 <= jj <= 88 and int(cb[0, ii, jj]) > -9990:
                    clat = -88.0 + 2.0 * jj
                    clon_r = 2.0 * ii
                    clon = clon_r - 360.0 if clon_r > 180.0 else clon_r
                    dkm = haversine_km(lat, lon, clat, clon)
                    if dkm <= 250.0 and dkm < best_d:
                        best_d = dkm
                        best = (ii, jj, clat, clon, dkm)
        return best

    def _window_weather(
        self, source: str, store, win: tuple[dt.date, dt.date] | None
    ) -> tuple[float, float]:
        """Return (temp_c, prcp_mm) for a single calendar window (start_date, end_date)."""
        if win is None:
            return float("nan"), float("nan")
        start, end = win
        if source == "Daymet_V4R1":
            expect = (end - start).days + 1
            vts, vps = [], []
            by_d = store["by_date"]
            for off in range(expect):
                rec = by_d.get(start + dt.timedelta(days=off))
                if rec is not None:
                    vts.append(rec[0])
                    vps.append(rec[1])
            if expect > 0 and len(vts) >= 0.8 * expect:
                return float(np.mean(vts)), (float(np.sum(vps)) if len(vps) >= 0.8 * expect else float("nan"))
            return float("nan"), float("nan")

        if source == "TerraClimate_V1.1":
            expect = end.month - start.month + 1
            vts, vps = [], []
            ym = store["ym"]
            for mo in range(start.month, end.month + 1):
                rec = ym.get((start.year, mo))
                if rec is not None:
                    vts.append(rec[0])
                    if rec[1] is not None:
                        vps.append(rec[1])
            if expect > 0 and len(vts) >= 0.8 * expect:
                return float(np.mean(vts)), (float(np.sum(vps)) if len(vps) >= 0.8 * expect else float("nan"))
            return float("nan"), float("nan")

        if source == "NOAA_ERSST_v5":
            expect = end.month - start.month + 1
            ii, jj = store
            cb = self.ersst_cube(start.year)
            if cb is None:
                return float("nan"), float("nan")
            vts = [int(cb[mo - 1, ii, jj]) * 0.01 for mo in range(start.month, end.month + 1) if int(cb[mo - 1, ii, jj]) > -9990]
            if expect > 0 and len(vts) >= 0.8 * expect:
                return float(np.mean(vts)), float("nan")
            return float("nan"), float("nan")

        if source == "NOAA_GHCNm_v4":
            sd = store
            vals = [sd[(start.year, m)] for m in range(start.month, end.month + 1) if (start.year, m) in sd]
            needed = 8 if (end.month - start.month == 11) else math.ceil(0.8 * (end.month - start.month + 1))
            if len(vals) >= needed:
                return float(np.mean(vals)), float("nan")
            return float("nan"), float("nan")

        return float("nan"), float("nan")

    def match_and_adjust_w1(
        self, s: dict, physical_calendar_lag1: bool = False
    ) -> dict:
        """Match series `s` to weather and run registered W1 regression adjustment.

        If `physical_calendar_lag1 == False`, reproduces `weather_continued.py`'s exact rule
        where step t-1 must also be a positive-host step (`int(t)-1 in pos`).
        If `physical_calendar_lag1 == True`, evaluates physical calendar step t-1 directly
        in the weather archive even when step t-1 was before k=0 or had zero/missing host count.
        """
        if s["environment_type"] == "laboratory":
            return {"matched": False, "reason": "laboratory"}
        lat, lon = float(s["latitude"]), float(s["longitude"])
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            return {"matched": False, "reason": "missing_coordinates"}

        pos_mask = s["values"] > 0.0
        yrs = s["step_years"][pos_mask]
        mos = s["step_months"][pos_mask]
        tt = s["steps"][pos_mask]
        iv = s["interval_class"]
        realm = s["realm_detail"]
        if int(np.sum((yrs >= 1700) & (yrs <= 2025))) < 15:
            return {"matched": False, "reason": "insufficient_calendar_dates"}

        source = None
        store = None
        if iv != "annual":
            opts = []
            for k_dm, v_dm in self.daymet.items():
                if haversine_km(lat, lon, v_dm["cell_lat"], v_dm["cell_lon"]) < 0.1:
                    opts.append((len(v_dm["by_date"]), k_dm, v_dm))
            if opts:
                _, _, store = max(opts, key=lambda x: x[0])
                source = "Daymet_V4R1"
        if source is None and iv == "subannual_biweekly_weekly":
            return {"matched": False, "reason": "monthly_weather_cannot_resolve_weekly_bin"}
        if source is None and realm == "Marine":
            cell = self.nearest_ersst(lat, lon, int(np.median(yrs[yrs >= 1700])))
            if cell is None:
                return {"matched": False, "reason": "no_ocean_grid_within_250km"}
            source = "NOAA_ERSST_v5"
            store = (cell[0], cell[1])
        if source is None:
            ii = max(1, min(4318, int(round((90.0 - lat) * 24.0 - 0.5))))
            jj = max(1, min(8638, int(round((lon + 180.0) * 24.0 - 0.5))))
            tc = self.terraclimate.get((ii, jj))
            if tc is not None and haversine_km(lat, lon, tc["cell_lat"], tc["cell_lon"]) <= 15.0:
                source = "TerraClimate_V1.1"
                store = tc

        n = len(tt)
        dates_iso = s.get("step_dates_iso")
        pos_idx = np.flatnonzero(pos_mask)
        spy = int(s["samples_per_year"])
        epoch = dt.date(1970, 1, 5)

        def window_for_pos_index(k: int) -> tuple[dt.date, dt.date] | None:
            y = int(yrs[k])
            mo = int(mos[k])
            if y < 1700 or y > 2025:
                return None
            if s["bin_days"] and dates_iso:
                d0 = dt.date.fromisoformat(dates_iso[int(pos_idx[k])])
                return d0, d0 + dt.timedelta(days=int(s["bin_days"]) - 1)
            if iv == "annual":
                return dt.date(y, 1, 1), dt.date(y, 12, 31)
            if s["database"] == "GPDD" and 1 <= int(s["subperiod"][pos_idx[k]]) <= 12:
                return dt.date(y, mo, 1), dt.date(y, mo, calendar.monthrange(y, mo)[1])
            if iv == "subannual_seasonal":
                w = 12 // spy
                m0 = ((max(1, mo) - 1) // w) * w + 1
                m1 = m0 + w - 1
                return dt.date(y, m0, 1), dt.date(y, m1, calendar.monthrange(y, m1)[1])
            if not (1 <= mo <= 12):
                return None
            return dt.date(y, mo, 1), dt.date(y, mo, calendar.monthrange(y, mo)[1])

        def prior_step_window(k: int) -> tuple[dt.date, dt.date] | None:
            """Calendar window one physical sampling step before positive-step index k."""
            cur_win = window_for_pos_index(k)
            if cur_win is None:
                return None
            s0, _ = cur_win
            if s["bin_days"]:
                bd = int(s["bin_days"])
                return s0 - dt.timedelta(days=bd), s0 - dt.timedelta(days=1)
            if iv == "annual":
                py = s0.year - 1
                return (dt.date(py, 1, 1), dt.date(py, 12, 31)) if py >= 1700 else None
            if iv == "subannual_seasonal":
                w = 12 // spy
                pm = s0.month - w
                py = s0.year
                if pm < 1:
                    pm += 12
                    py -= 1
                if py < 1700:
                    return None
                pm_end = pm + w - 1
                return dt.date(py, pm, 1), dt.date(py, pm_end, calendar.monthrange(py, pm_end)[1])
            # monthly
            pm = s0.month - 1
            py = s0.year
            if pm < 1:
                pm = 12
                py -= 1
            if py < 1700:
                return None
            return dt.date(py, pm, 1), dt.date(py, pm, calendar.monthrange(py, pm)[1])

        T_cur = np.full(n, np.nan, dtype=float)
        P_cur = np.full(n, np.nan, dtype=float)
        if source is not None:
            for k in range(n):
                T_cur[k], P_cur[k] = self._window_weather(source, store, window_for_pos_index(k))

        if int(np.sum(np.isfinite(T_cur))) < 15 and realm != "Marine" and iv != "subannual_biweekly_weekly":
            # GHCN-Monthly station fallback within <= 250 km
            cands = sorted(
                (haversine_km(lat, lon, la, lo), sid, la, lo)
                for sid, la, lo in self.ghcn_stations
                if abs(la - lat) <= 2.5
            )
            cand_slice = cands[: self.ghcn_max_rank] if self.ghcn_max_rank > 0 else cands
            for dkm, sid, _, _ in cand_slice:
                if dkm > 250.0:
                    continue
                sd = self.ghcn_monthly.get(sid, {})
                t_try = np.array([self._window_weather("NOAA_GHCNm_v4", sd, window_for_pos_index(k))[0] for k in range(n)])
                if int(np.sum(np.isfinite(t_try))) >= 15:
                    T_cur = t_try
                    P_cur[:] = np.nan
                    source = "NOAA_GHCNm_v4"
                    store = sd
                    break

        ok = np.isfinite(T_cur)
        if int(np.sum(ok)) < 15:
            return {"matched": False, "reason": "fewer_than_15_matched_steps", "source": source}

        pos_map = {int(v): idx for idx, v in enumerate(tt)}
        if not physical_calendar_lag1:
            cur_idx = np.array(
                [i for i, ts in enumerate(tt) if ok[i] and (int(ts) - 1) in pos_map and ok[pos_map[int(ts) - 1]]],
                dtype=int,
            )
            if len(cur_idx) < 15:
                return {"matched": False, "reason": "fewer_than_15_current_lag_pairs", "source": source}
            lag_idx = np.array([pos_map[int(tt[i]) - 1] for i in cur_idx], dtype=int)
            t_lag_vals = T_cur[lag_idx]
        else:
            T_prev = np.array([
                T_cur[pos_map[int(tt[i]) - 1]]
                if ((int(tt[i]) - 1) in pos_map and ok[pos_map[int(tt[i]) - 1]])
                else self._window_weather(source, store, prior_step_window(i))[0]
                for i in range(n)
            ])
            cur_idx = np.array([i for i in range(n) if ok[i] and math.isfinite(T_prev[i])], dtype=int)
            if len(cur_idx) < 15:
                return {"matched": False, "reason": "fewer_than_15_current_lag_pairs", "source": source}
            t_lag_vals = T_prev[cur_idx]

        t_use = tt[cur_idx].astype(float)
        if int(np.sum(np.diff(t_use) == 1.0)) < 12:
            return {"matched": False, "reason": "fewer_than_12_adjacent_residual_pairs", "source": source}

        y_use = np.log(s["values"][pos_mask][cur_idx])
        tc_use = T_cur[cur_idx]
        pc_use = P_cur[cur_idx]

        cols = [np.ones(len(t_use)), t_use - float(np.mean(t_use)), tc_use - float(np.mean(tc_use)), t_lag_vals - float(np.mean(t_lag_vals))]
        use_p = bool(np.all(np.isfinite(pc_use)) and float(np.std(pc_use)) > 1e-10)
        if use_p:
            cols.append(pc_use - float(np.mean(pc_use)))
        X = np.column_stack(cols)
        Q = cm.orthonormal_columns(X)
        res = y_use - Q @ (Q.T @ y_use)

        m_pair = cm.estimate_series_quantities(t_use, y_use, detrend=True)
        m_pair["mu0_r1"] = cm.wn_null_r1_linear(t_use)
        m_pair["delta_r1"] = m_pair["r1"] - m_pair["mu0_r1"]
        m_pair["r1_bc"] = cm.calibrate_ar1_phi(t_use, m_pair["r1"])

        m_adj = cm.estimate_series_quantities(t_use, res, detrend=True)
        m_adj["sigma2"] = float((res @ res) / float(len(t_use) - Q.shape[1]))
        m_adj["sigma"] = math.sqrt(max(m_adj["sigma2"], 0.0))
        m_adj["mu0_r1"] = cm.wn_null_r1_projected(t_use, Q)
        m_adj["delta_r1"] = m_adj["r1"] - m_adj["mu0_r1"]
        m_adj["r1_bc"] = cm.calibrate_ar1_phi(t_use, m_adj["r1"], Q=Q)

        return {
            "matched": True,
            "source": source,
            "paired_metrics": m_pair,
            "adjusted_metrics": m_adj,
        }
