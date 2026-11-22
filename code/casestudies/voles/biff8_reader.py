"""Pure-Python BIFF8 (.xls) reader using the Python standard library only.

Reads OLE2 Compound Document streams (`Workbook` / `Book`) and parses BIFF8
records:
  - 0x0085 BOUNDSHEET
  - 0x00FC SST (Shared String Table, including multi-record 0x003C CONTINUE)
  - 0x0200 DIMENSIONS
  - 0x00FD LABELSST (string cell from SST)
  - 0x0204 LABEL (inline string cell)
  - 0x0203 NUMBER (IEEE 754 float64 cell)
  - 0x027E RK (compressed integer / float cell)
  - 0x00BD MULRK (multiple consecutive RK cells in one row)
  - 0x0006 FORMULA (cached numeric result, or followed by 0x0207 STRING)
  - 0x0205 BOOLERR (boolean or error cell)

Verified against known values before use (study rules Rule 7).
"""
from __future__ import annotations

import datetime
import struct
from typing import Any


def _decode_rk(rk: int) -> float | int:
    is_mul100 = bool(rk & 0x01)
    is_int = bool(rk & 0x02)
    if is_int:
        # Signed 30-bit integer
        signed_rk = rk if rk < 0x80000000 else rk - 0x100000000
        val: float | int = signed_rk >> 2
        if is_mul100:
            val = val / 100.0
    else:
        raw_u64 = (rk & 0xFFFFFFFC) << 32
        val = struct.unpack("<d", struct.pack("<Q", raw_u64))[0]
        if is_mul100:
            val = val / 100.0
    return val


def excel_serial_to_date(serial: float) -> datetime.date:
    """Convert Excel 1900-system serial date (with 1900 leap-year bug, serial>=61)
    to `datetime.date`. Checked: 37370 -> 2002-04-24, 39945 -> 2009-05-12.
    """
    days = int(round(float(serial)))
    if days >= 60:
        return datetime.date(1899, 12, 30) + datetime.timedelta(days=days)
    return datetime.date(1899, 12, 31) + datetime.timedelta(days=days)


def read_biff8_workbook(path: str) -> dict[str, list[list[Any]]]:
    """Return `{sheet_name: rows}` where `rows` is a dense 2D list of Python values
    (`None`, `str`, `float`, `int`, `bool`).
    """
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
        raise ValueError(f"{path}: not an OLE2 compound document")

    sec_shift = struct.unpack_from("<H", data, 30)[0]
    sec_size = 1 << sec_shift
    dir_start = struct.unpack_from("<I", data, 48)[0]
    difat_first = struct.unpack_from("<I", data, 68)[0]
    difat_cnt = struct.unpack_from("<I", data, 72)[0]

    difat = list(struct.unpack_from("<109I", data, 76))
    s = difat_first
    for _ in range(difat_cnt):
        if s >= 0xFFFFFFFE:
            break
        off = 512 + s * sec_size
        vals = struct.unpack_from(f"<{sec_size // 4}I", data, off)
        difat.extend(vals[:-1])
        s = vals[-1]

    fat: list[int] = []
    for fs in difat:
        if fs >= 0xFFFFFFFE:
            continue
        off = 512 + fs * sec_size
        fat.extend(struct.unpack_from(f"<{sec_size // 4}I", data, off))

    def read_chain(start_sec: int) -> bytes:
        out = bytearray()
        cur = start_sec
        seen = set()
        while cur < 0xFFFFFFFE and cur not in seen:
            seen.add(cur)
            off = 512 + cur * sec_size
            out.extend(data[off : off + sec_size])
            cur = fat[cur]
        return bytes(out)

    dir_bytes = read_chain(dir_start)
    wb_stream: bytes | None = None
    for i in range(0, len(dir_bytes), 128):
        entry = dir_bytes[i : i + 128]
        if len(entry) < 128:
            break
        nlen = struct.unpack_from("<H", entry, 64)[0]
        if nlen <= 2:
            continue
        name = entry[: nlen - 2].decode("utf-16le", "replace")
        if name in ("Workbook", "Book"):
            start_sec = struct.unpack_from("<I", entry, 116)[0]
            size = struct.unpack_from("<I", entry, 120)[0]
            wb_stream = read_chain(start_sec)[:size]
            break

    if wb_stream is None:
        raise ValueError(f"{path}: no Workbook/Book stream found")

    # Pass 1: Globals (BOUNDSHEET + SST)
    pos = 0
    sheets: list[tuple[str, int]] = []
    sst: list[str] = []
    while pos + 4 <= len(wb_stream):
        rt, rl = struct.unpack_from("<HH", wb_stream, pos)
        rec = wb_stream[pos + 4 : pos + 4 + rl]
        pos += 4 + rl
        if rt == 0x0085:  # BOUNDSHEET
            bof_pos = struct.unpack_from("<I", rec, 0)[0]
            slen = rec[6]
            flags = rec[7]
            if flags & 0x01:
                sname = rec[8 : 8 + 2 * slen].decode("utf-16le", "replace")
            else:
                sname = rec[8 : 8 + slen].decode("latin1", "replace")
            sheets.append((sname, bof_pos))
        elif rt == 0x00FC:  # SST
            chunks = [rec]
            while pos + 4 <= len(wb_stream):
                nrt, nrl = struct.unpack_from("<HH", wb_stream, pos)
                if nrt == 0x003C:  # CONTINUE
                    chunks.append(wb_stream[pos + 4 : pos + 4 + nrl])
                    pos += 4 + nrl
                else:
                    break
            cidx = 0
            coff = 8
            uniq = struct.unpack_from("<I", chunks[0], 4)[0]

            def need_byte() -> int:
                nonlocal cidx, coff
                while cidx < len(chunks) and coff >= len(chunks[cidx]):
                    cidx += 1
                    coff = 0
                b = chunks[cidx][coff]
                coff += 1
                return b

            def read_n(n_bytes: int) -> bytes:
                nonlocal cidx, coff
                out_b = bytearray()
                while len(out_b) < n_bytes and cidx < len(chunks):
                    avail = len(chunks[cidx]) - coff
                    if avail == 0:
                        cidx += 1
                        coff = 0
                        continue
                    take = min(n_bytes - len(out_b), avail)
                    out_b.extend(chunks[cidx][coff : coff + take])
                    coff += take
                return bytes(out_b)

            for _ in range(uniq):
                if cidx >= len(chunks):
                    break
                hdr = read_n(2)
                if len(hdr) < 2:
                    break
                cch = struct.unpack_from("<H", hdr, 0)[0]
                flg = need_byte()
                f_high = bool(flg & 0x01)
                f_ext = bool(flg & 0x04)
                f_rich = bool(flg & 0x08)
                rt_runs = struct.unpack_from("<H", read_n(2), 0)[0] if f_rich else 0
                ext_sz = struct.unpack_from("<I", read_n(4), 0)[0] if f_ext else 0
                chars: list[str] = []
                rem = cch
                while rem > 0 and cidx < len(chunks):
                    avail = len(chunks[cidx]) - coff
                    if avail == 0:
                        cidx += 1
                        coff = 0
                        if cidx < len(chunks):
                            f_high = bool(chunks[cidx][0] & 0x01)
                            coff = 1
                        continue
                    if f_high:
                        take_c = min(rem, avail // 2)
                        raw_c = chunks[cidx][coff : coff + 2 * take_c]
                        coff += 2 * take_c
                        chars.append(raw_c.decode("utf-16le", "replace"))
                        rem -= take_c
                    else:
                        take_c = min(rem, avail)
                        raw_c = chunks[cidx][coff : coff + take_c]
                        coff += take_c
                        chars.append(raw_c.decode("latin1", "replace"))
                        rem -= take_c
                if rt_runs:
                    read_n(4 * rt_runs)
                if ext_sz:
                    read_n(ext_sz)
                sst.append("".join(chars))
        elif rt == 0x000A:  # EOF of workbook globals
            break

    # Pass 2: Individual Worksheets
    result: dict[str, list[list[Any]]] = {}
    for sname, bof_pos in sheets:
        p = bof_pos
        cells: dict[tuple[int, int], Any] = {}
        max_r = -1
        max_c = -1
        depth = 0
        last_formula_rc: tuple[int, int] | None = None

        while p + 4 <= len(wb_stream):
            rt, rl = struct.unpack_from("<HH", wb_stream, p)
            rec = wb_stream[p + 4 : p + 4 + rl]
            p += 4 + rl
            if rt == 0x0809:  # BOF
                depth += 1
                continue
            elif rt == 0x000A:  # EOF
                depth -= 1
                if depth <= 0:
                    break
                continue
            if depth != 1:
                continue

            if rt == 0x00FD and len(rec) >= 10:  # LABELSST
                rw, cl, _, ssti = struct.unpack_from("<HHHI", rec, 0)
                val = sst[ssti] if ssti < len(sst) else None
                cells[(rw, cl)] = val
                max_r = max(max_r, rw)
                max_c = max(max_c, cl)
                last_formula_rc = None
            elif rt == 0x0204 and len(rec) >= 8:  # LABEL
                rw, cl, _, slen = struct.unpack_from("<HHHH", rec, 0)
                if len(rec) > 8 and rl > 8 + slen:
                    flg = rec[8]
                    txt = rec[9 : 9 + 2 * slen].decode("utf-16le", "replace") if (flg & 1) else rec[9 : 9 + slen].decode("latin1", "replace")
                else:
                    txt = rec[8 : 8 + slen].decode("latin1", "replace")
                cells[(rw, cl)] = txt
                max_r = max(max_r, rw)
                max_c = max(max_c, cl)
                last_formula_rc = None
            elif rt == 0x0203 and len(rec) >= 14:  # NUMBER
                rw, cl, _, dval = struct.unpack_from("<HHHd", rec, 0)
                cells[(rw, cl)] = dval
                max_r = max(max_r, rw)
                max_c = max(max_c, cl)
                last_formula_rc = None
            elif rt == 0x027E and len(rec) >= 10:  # RK
                rw, cl, _, rk = struct.unpack_from("<HHHI", rec, 0)
                cells[(rw, cl)] = _decode_rk(rk)
                max_r = max(max_r, rw)
                max_c = max(max_c, cl)
                last_formula_rc = None
            elif rt == 0x00BD and len(rec) >= 6:  # MULRK
                rw, first_c = struct.unpack_from("<HH", rec, 0)
                last_c = struct.unpack_from("<H", rec, len(rec) - 2)[0]
                n_rk = last_c - first_c + 1
                for k in range(n_rk):
                    off_k = 4 + 6 * k
                    if off_k + 6 <= len(rec) - 2:
                        _, rk = struct.unpack_from("<HI", rec, off_k)
                        cl = first_c + k
                        cells[(rw, cl)] = _decode_rk(rk)
                        max_r = max(max_r, rw)
                        max_c = max(max_c, cl)
                last_formula_rc = None
            elif rt == 0x0006 and len(rec) >= 14:  # FORMULA
                rw, cl, _ = struct.unpack_from("<HHH", rec, 0)
                res_bytes = rec[6:14]
                if res_bytes[6:8] == b"\xff\xff":
                    rtype = res_bytes[0]
                    if rtype == 0:
                        # String result follows in a 0x0207 STRING record
                        last_formula_rc = (rw, cl)
                    elif rtype == 1:
                        cells[(rw, cl)] = bool(res_bytes[2])
                        last_formula_rc = None
                    else:
                        cells[(rw, cl)] = None
                        last_formula_rc = None
                else:
                    cells[(rw, cl)] = struct.unpack("<d", res_bytes)[0]
                    last_formula_rc = None
                max_r = max(max_r, rw)
                max_c = max(max_c, cl)
            elif rt == 0x0207 and last_formula_rc is not None and len(rec) >= 3:  # STRING
                slen = struct.unpack_from("<H", rec, 0)[0]
                flg = rec[2]
                if flg & 1:
                    sval = rec[3 : 3 + 2 * slen].decode("utf-16le", "replace")
                else:
                    sval = rec[3 : 3 + slen].decode("latin1", "replace")
                cells[last_formula_rc] = sval
                last_formula_rc = None
            elif rt == 0x0205 and len(rec) >= 8:  # BOOLERR
                rw, cl, _, bval, is_err = struct.unpack_from("<HHHBB", rec, 0)
                cells[(rw, cl)] = None if is_err else bool(bval)
                max_r = max(max_r, rw)
                max_c = max(max_c, cl)
                last_formula_rc = None

        if max_r < 0 or max_c < 0:
            result[sname] = []
        else:
            grid = [[None] * (max_c + 1) for _ in range(max_r + 1)]
            for (r, c), v in cells.items():
                grid[r][c] = v
            result[sname] = grid

    return result
