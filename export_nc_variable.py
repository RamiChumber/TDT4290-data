#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Headless plotting
import matplotlib
matplotlib.use("Agg")

from pathlib import Path
import argparse
import os
import numpy as np
import netCDF4 as nc
import cv2 as cv
from matplotlib import pyplot as plt

# -- Hjelpefunksjoner ---------------------------------------------------------

def _read_variable(nc_path: Path, var_name: str):
    """Åpner NetCDF og returnerer (dataset, var_data, dims, coords_dict)."""
    ds = nc.Dataset(nc_path, "r")
    if var_name not in ds.variables:
        ds.close()
        raise KeyError(f"Fant ikke variabel '{var_name}' i {nc_path.name}. "
                       f"Tilgjengelige variabler: {list(ds.variables.keys())[:20]}...")
    var = ds.variables[var_name]
    data = var[:]  # lazily loads via netCDF4; converts on slicing
    dims = var.dimensions  # tuple med dimensjonsnavn
    coords = {d: ds.variables[d][:] for d in dims if d in ds.variables}
    return ds, np.array(data), dims, coords

def _infer_time_coord(ds, coords):
    """Prøv å hente 'time' koordinat + (valgfritt) konvertering til sekunder/int."""
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

    # Prøv å tolke CF-units -> int labels (fallback: råverdier)
    units = getattr(tvar, "units", None)
    calendar = getattr(tvar, "calendar", "standard")
    try:
        if units:
            # num2date -> datetime-objekter -> ISO strings / posix
            from netCDF4 import num2date
            dts = num2date(tvals, units, calendar=calendar)
            # Label som YYYYmmddHHMMSS for filnavn (ingen kolon / mellomrom)
            labels = [f"{dt.year:04d}{dt.month:02d}{dt.day:02d}"
                      f"{dt.hour:02d}{dt.minute:02d}{dt.second:02d}" for dt in np.atleast_1d(dts)]
            return np.array(labels), time_var_name
        else:
            return np.array(tvals).astype(str), time_var_name
    except Exception:
        # Fallback: bruk rå tidsverdier
        return np.array(tvals).astype(str), time_var_name

def _ensure_output_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _make_uint8(image: np.ndarray):
    """Skalerer til 0-255 uint8 for trygg lagring med imsave/cv.imwrite."""
    arr = np.asarray(image, dtype=float)
    mmin = np.nanmin(arr)
    mmax = np.nanmax(arr)
    if not np.isfinite(mmin) or not np.isfinite(mmax) or mmax == mmin:
        return np.zeros_like(arr, dtype=np.uint8)
    norm = (arr - mmin) / (mmax - mmin)
    return (norm * 255.0).clip(0, 255).astype(np.uint8)

def _resample2d(field2d: np.ndarray, factor: float, method: str = "cubic") -> np.ndarray:
    """Skaler 2D felt med OpenCV. method: 'nearest'|'linear'|'cubic'."""
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
    """Valgfri blur med Gaussian kernel (mer naturlig enn box)."""
    if blur_kernel is None or blur_kernel <= 1:
        return field2d
    # kernel må være oddetall
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
      - 2D: (y, x) -> lagrer én fil (uten tidslabel)
      - 3D: (time, y, x)
      - 4D: (time, level, y, x) -> bruk 'level_index' for å velge nivå

    Returnerer liste over lagrede filstier.
    """
    nc_path = Path(nc_path)
    out_dir = Path(out_dir)
    _ensure_output_dir(out_dir)

    ds, data, dims, coords = _read_variable(nc_path, variable)
    try:
        # Finn tidslabels (om de finnes)
        time_labels, time_dim = _infer_time_coord(ds, coords)

        saved = []

        if data.ndim == 2:
            # (y, x)
            field = np.array(data, dtype=float)
            field = _resample2d(field, scale_factor, interp)
            field = np.flipud(field)  # matchende orientering som før
            field = _optional_blur(field, blur_kernel)

            img = _make_uint8(field)
            out_path = out_dir / f"{variable}.png"

            # Bruk matplotlib for konsistent colormap og dpi
            plt.figure()
            plt.axis("off")
            plt.imshow(img, cmap=cmap)
            plt.savefig(out_path, bbox_inches="tight", pad_inches=0, dpi=dpi)
            plt.close()
            saved.append(out_path)

        elif data.ndim == 3:
            # Antas (time, y, x) eller (y, x, time). Prøv å lese av dims.
            # Foretrekk CF-rekkefølge via dims.
            if time_dim and dims.index(time_dim) == 0:
                # (time, y, x)
                for ti in range(data.shape[0]):
                    field = np.array(data[ti, :, :], dtype=float)
                    field = _resample2d(field, scale_factor, interp)
                    field = np.flipud(field)
                    field = _optional_blur(field, blur_kernel)

                    img = _make_uint8(field)
                    label = time_labels[ti] if time_labels is not None else f"{ti:04d}"
                    out_path = out_dir / f"{variable}_{label}.png"
                    plt.figure()
                    plt.axis("off")
                    plt.imshow(img, cmap=cmap)
                    plt.savefig(out_path, bbox_inches="tight", pad_inches=0, dpi=dpi)
                    plt.close()
                    saved.append(out_path)
            else:
                # Fallback: anta (time, y, x) uansett
                for ti in range(data.shape[0]):
                    field = np.array(data[ti, :, :], dtype=float)
                    field = _resample2d(field, scale_factor, interp)
                    field = np.flipud(field)
                    field = _optional_blur(field, blur_kernel)

                    img = _make_uint8(field)
                    label = f"{ti:04d}" if time_labels is None else time_labels[ti]
                    out_path = out_dir / f"{variable}_{label}.png"
                    plt.figure()
                    plt.axis("off")
                    plt.imshow(img, cmap=cmap)
                    plt.savefig(out_path, bbox_inches="tight", pad_inches=0, dpi=dpi)
                    plt.close()
                    saved.append(out_path)

        elif data.ndim == 4:
            # Forventer (time, level, y, x)
            tlen, llen = data.shape[0], data.shape[1]
            if not (0 <= level_index < llen):
                raise IndexError(f"level_index={level_index} utenfor [0, {llen-1}]")
            for ti in range(tlen):
                field = np.array(data[ti, level_index, :, :], dtype=float)
                field = _resample2d(field, scale_factor, interp)
                field = np.flipud(field)
                field = _optional_blur(field, blur_kernel)

                img = _make_uint8(field)
                label = time_labels[ti] if time_labels is not None else f"{ti:04d}"
                out_path = out_dir / f"{variable}_lev{level_index}_{label}.png"
                plt.figure()
                plt.axis("off")
                plt.imshow(img, cmap=cmap)
                plt.savefig(out_path, bbox_inches="tight", pad_inches=0, dpi=dpi)
                plt.close()
                saved.append(out_path)
        else:
            raise ValueError(f"Variabel '{variable}' har ikke-støttet dimensjon {data.shape}.")

        return saved

    finally:
        ds.close()

# -- Valgfri rask forhåndsvisning (for debugging/QA) --------------------------

def preview_frame(nc_path: str | Path, variable: str, time_index: int = 0, level_index: int = 0):
    """
    Viser (ikke lagrer) et enkelt tidssteg ved hjelp av Matplotlib.
    Nyttig for rask QA. Krever interaktiv backend; ikke bruk på headless server.
    """
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

        plt.figure()
        plt.imshow(frame, cmap="gray")
        plt.title(f"{variable} (sample)")
        plt.colorbar()
        plt.show()
    finally:
        ds.close()

# -- CLI ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Eksporter NetCDF-variabel til PNG-sekvens.")
    p.add_argument("nc_path", type=str, help="Sti til .nc filen")
    p.add_argument("variable", type=str, help="Variabelnavn i filen (f.eks. 'temperature')")
    p.add_argument("out_dir", type=str, help="Mappe for utdata (opprettes ved behov)")
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
    saved = export_variable_to_pngs(
        nc_path=args.nc_path,
        variable=args.variable,
        out_dir=args.out_dir,
        scale_factor=args.scale,
        interp=args.interp,
        blur_kernel=args.blur,
        level_index=args.level,
        cmap=args.cmap,
        dpi=args.dpi,
    )
    print(f"Lagret {len(saved)} filer i '{args.out_dir}'.")

if __name__ == "__main__":
    main()
