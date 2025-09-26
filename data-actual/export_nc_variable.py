#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Headless plotting
import matplotlib
matplotlib.use("Agg")

from pathlib import Path
import argparse
import sys
import numpy as np
import netCDF4 as nc
import cv2 as cv
from matplotlib import pyplot as plt

# ---------- QUICK-START DEFAULTS (din filsti her) ----------
DEFAULTS = dict(
    nc_path=Path("/Users/karestormark/Documents/H25/Kundestyrt/TDT4290-data/data-actual/dp_3d.001.nc"),
    variable="q",              # <-- bytt til riktig variabelnavn om nødvendig
    out_dir=Path("out_imgs"),     # lagres i prosjektmappa
    scale=2.0,
    interp="cubic",
    blur=None,                    # f.eks. 21 for Gaussian blur
    level=0,
    cmap="gray",
    dpi=300,
)
# -----------------------------------------------------------


# -- Hjelpefunksjoner ---------------------------------------------------------

def _read_variable(nc_path: Path, var_name: str):
    """Åpner NetCDF og returnerer (dataset, var_data, dims, coords_dict)."""
    ds = nc.Dataset(nc_path, "r")
    if var_name not in ds.variables:
        try:
            avail = list(ds.variables.keys())
        finally:
            pass
        ds.close()
        raise KeyError(
            f"Fant ikke variabel '{var_name}' i {nc_path.name}.\n"
            f"Tilgjengelige variabler (første 20): {avail[:20]}"
        )
    var = ds.variables[var_name]
    data = var[:]  # lazily loads via netCDF4; converts on slicing
    dims = var.dimensions  # tuple med dimensjonsnavn
    coords = {d: ds.variables[d][:] for d in dims if d in ds.variables}
    return ds, np.array(data), dims, coords

def _infer_time_coord(ds, coords):
    """Prøv å hente 'time' koordinat + (valgfritt) konvertering til label-strenger."""
    time_var_name = None
    for cand in ("time", "Time", "t"):
        if cand in coords:
            time_var_name = cand
            break
    if time_var_name is None and "time" in ds.variables:
        time_var_name = "time"
        coords["time"] = ds.variables["time"][:]

    if time_var_name is None:
        return None, None  # ikke tidsdimensjon

    tvar = ds.variables[time_var_name]
    tvals = coords[time_var_name]

    # Prøv å tolke CF-units -> ISO labels (fallback: råverdier)
    units = getattr(tvar, "units", None)
    calendar = getattr(tvar, "calendar", "standard")
    try:
        if units:
            from netCDF4 import num2date
            dts = num2date(tvals, units, calendar=calendar)
            labels = [f"{dt.year:04d}{dt.month:02d}{dt.day:02d}"
                      f"{dt.hour:02d}{dt.minute:02d}{dt.second:02d}" for dt in np.atleast_1d(dts)]
            return np.array(labels), time_var_name
        else:
            return np.array(tvals).astype(str), time_var_name
    except Exception:
        return np.array(tvals).astype(str), time_var_name

def _ensure_output_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _make_uint8(image: np.ndarray):
    """Skalerer til 0-255 uint8 for trygg lagring."""
    arr = np.asarray(image, dtype=float)
    mmin = np.nanmin(arr)
    mmax = np.nanmax(arr)
    if not np.isfinite(mmin) or not np.isfinite(mmax) or mmax == mmin:
        return np.zeros_like(arr, dtype=np.uint8)
    norm = (arr - mmin) / (mmax - mmin)
    return (norm * 255.0).clip(0, 255).astype(np.uint8)

def _resample2d(field2d: np.ndarray, factor: float, method: str = "cubic") -> np.ndarray:
    """Skaler 2D felt med OpenCV. method: 'nearest'|'linear'|'cubic'|'area'|'lanczos'."""
    if factor == 1 or factor <= 0:
        return field2d
    interp_map = {
        "nearest": cv.INTER_NEAREST,
        "linear":  cv.INTER_LINEAR,
        "cubic":   cv.INTER_CUBIC,
        "area":    cv.INTER_AREA,
        "lanczos": cv.INTER_LANCZOS4,
    }
    inter = interp_map.get(method.lower(), cv.INTER_CUBIC)
    h, w = field2d.shape
    new_w = max(1, int(round(w * factor)))
    new_h = max(1, int(round(h * factor)))
    return cv.resize(field2d, (new_w, new_h), interpolation=inter)

def _optional_blur(field2d: np.ndarray, blur_kernel: int | None):
    """Valgfri blur med Gaussian kernel."""
    if blur_kernel is None or blur_kernel <= 1:
        return field2d
    k = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
    return cv.GaussianBlur(field2d, (k, k), 0)

# -- Kjernefunksjon -----------------------------------------------------------

def export_variable_to_pngs(
    nc_path: str | Path,
    variable: str,
    out_dir: str | Path,
    scale_factor: float = 2.0,
    interp: str = "cubic",
    blur_kernel: int | None = None,
    level_index: int = 0,
    cmap: str = "gray",
    dpi: int = 300
) -> list[Path]:
    """
    Leser en NetCDF-variabel og lagrer én PNG per tidssteg.

    Støttede former:
      - 2D: (y, x)
      - 3D: (time, y, x)
      - 4D: (time, level, y, x) -> bruk 'level_index'
    """
    nc_path = Path(nc_path)
    out_dir = Path(out_dir)
    _ensure_output_dir(out_dir)

    ds, data, dims, coords = _read_variable(nc_path, variable)
    try:
        time_labels, time_dim = _infer_time_coord(ds, coords)
        saved = []

        def save_img(field2d, out_path):
            field = _resample2d(np.array(field2d, dtype=float), scale_factor, interp)
            field = np.flipud(field)
            field = _optional_blur(field, blur_kernel)
            img = _make_uint8(field)
            plt.figure()
            plt.axis("off")
            plt.imshow(img, cmap=cmap)
            plt.savefig(out_path, bbox_inches="tight", pad_inches=0, dpi=dpi)
            plt.close()

        if data.ndim == 2:
            out_path = out_dir / f"{variable}.png"
            save_img(data, out_path)
            saved.append(out_path)

        elif data.ndim == 3:
            # anta (time, y, x)
            for ti in range(data.shape[0]):
                label = time_labels[ti] if time_labels is not None else f"{ti:04d}"
                out_path = out_dir / f"{variable}_{label}.png"
                save_img(data[ti, :, :], out_path)
                saved.append(out_path)

        elif data.ndim == 4:
            # (time, level, y, x)
            tlen, llen = data.shape[0], data.shape[1]
            if not (0 <= level_index < llen):
                raise IndexError(f"level_index={level_index} utenfor [0, {llen-1}]")
            for ti in range(tlen):
                label = time_labels[ti] if time_labels is not None else f"{ti:04d}"
                out_path = out_dir / f"{variable}_lev{level_index}_{label}.png"
                save_img(data[ti, level_index, :, :], out_path)
                saved.append(out_path)
        else:
            raise ValueError(f"Variabel '{variable}' har ikke-støttet dimensjon {data.shape}.")

        return saved
    finally:
        ds.close()

# -- Valgfri rask forhåndsvisning --------------------------------------------

def preview_frame(nc_path: str | Path, variable: str, time_index: int = 0, level_index: int = 0):
    import matplotlib.pyplot as plt
    ds, data, dims, coords = _read_variable(Path(nc_path), variable)
    try:
        if data.ndim == 2:
            frame = data
        elif data.ndim == 3:
            frame = data[time_index, :, :]
        elif data.ndim == 4:
            frame = data[time_index, level_index, :, :]
        else:
            raise ValueError(f"Ukjent shape: {data.shape}")
        plt.figure(); plt.imshow(frame, cmap="gray"); plt.title(f"{variable} (sample)")
        plt.colorbar(); plt.show()
    finally:
        ds.close()

# -- CLI ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Eksporter NetCDF-variabel til PNG-sekvens.")
    p.add_argument("nc_path", nargs="?", type=str, help="Sti til .nc-filen")
    p.add_argument("variable", nargs="?", type=str, help="Variabelnavn i filen")
    p.add_argument("out_dir", nargs="?", type=str, help="Mappe for utdata (opprettes ved behov)")
    p.add_argument("--scale", type=float, default=2.0, help="Oppskaleringsfaktor (default 2.0)")
    p.add_argument("--interp", type=str, default="cubic",
                   choices=["nearest", "linear", "cubic", "area", "lanczos"],
                   help="Interpolasjonsmetode for resampling")
    p.add_argument("--blur", type=int, default=None,
                   help="Valgfri Gaussian blur kernel-størrelse (oddetall). Eks: 21")
    p.add_argument("--level", type=int, default=0, help="Level-indeks for 4D variabler (default 0)")
    p.add_argument("--cmap", type=str, default="gray", help="Matplotlib colormap for PNG-ene")
    p.add_argument("--dpi", type=int, default=300, help="DPI ved lagring")
    return p.parse_args()

def main():
    args = parse_args()

    # Hvis ingen CLI-argumenter er gitt, bruk QUICK-START DEFAULTS
    if args.nc_path is None and args.variable is None and args.out_dir is None:
        nc_path = DEFAULTS["nc_path"]
        variable = DEFAULTS["variable"]
        out_dir = DEFAULTS["out_dir"]
        scale = DEFAULTS["scale"]
        interp = DEFAULTS["interp"]
        blur = DEFAULTS["blur"]
        level = DEFAULTS["level"]
        cmap = DEFAULTS["cmap"]
        dpi = DEFAULTS["dpi"]
    else:
        nc_path = args.nc_path
        variable = args.variable
        out_dir = args.out_dir
        scale = args.scale
        interp = args.interp
        blur = args.blur
        level = args.level
        cmap = args.cmap
        dpi = args.dpi

    saved = export_variable_to_pngs(
        nc_path=nc_path,
        variable=variable,
        out_dir=out_dir,
        scale_factor=scale,
        interp=interp,
        blur_kernel=blur,
        level_index=level,
        cmap=cmap,
        dpi=dpi,
    )
    print(f"Lagret {len(saved)} filer i '{out_dir}'.")

if __name__ == "__main__":
    main()
