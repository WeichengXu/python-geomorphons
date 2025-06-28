#!/usr/bin/env python3
"""
Geomorphons (Python)
====================
An exact Python version of the WhiteboxTools `Geomorphons` algorithm,
classifying landform elements based on elevation patterns in a DEM.

Implements:
- Full raster processing (`geomorphons`)
- Single-cell classification (`geomorphon_of_cell`)

Requirements:
- numpy >= 1.20
- rasterio >= 1.3

Usage:
-------
From the command line:

    python geomorphons.py \
        --dem dem.tif \
        --output geomorphons.tif \
        --search 50 \
        --threshold 0.0 \
        --fdist 0 \
        --skip 0 \
        --forms \
        --residuals

Author:
--------
Weicheng Xu  
University of North Carolina at Chapel Hill  
Created: June 28, 2025
"""


import argparse
import math
import sys
import time
from pathlib import Path
from typing import Union

import numpy as np
import rasterio
from numpy.linalg import lstsq
from rasterio import Affine

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

CLASSES_10 = np.array([
    [1, 1, 1, 8, 8, 9, 9, 9, 10],
    [1, 1, 8, 8, 8, 9, 9, 9,  0],
    [1, 4, 6, 6, 7, 7, 9, 0,  0],
    [4, 4, 6, 6, 6, 7, 0, 0,  0],
    [4, 4, 5, 6, 6, 0, 0, 0,  0],
    [3, 3, 5, 5, 0, 0, 0, 0,  0],
    [3, 3, 3, 0, 0, 0, 0, 0,  0],
    [3, 3, 0, 0, 0, 0, 0, 0,  0],
    [2, 0, 0, 0, 0, 0, 0, 0,  0],
], dtype=np.int16)

def _generate_gtc() -> np.ndarray:
    """Return the minimal rotation/reflection code for each ternary pattern."""
    gtc = np.full(3 ** 8, 65535, dtype=np.uint16)
    tern, rtern, tmp, tmpr = [0] * 8, [0] * 8, [0] * 8, [0] * 8

    for value in range(6561):
        v = value
        for i in range(8):
            tern[i] = v % 3
            rtern[7 - i] = tern[i]
            v //= 3

        best = 6561
        for shift in range(8):
            code = rcode = 0
            for i in range(8):
                tmp[i] = tern[(i - shift) % 8]
                tmpr[i] = rtern[(i - shift) % 8]
            for i in range(8):
                code += tmp[i] * (3 ** i)
                rcode += tmpr[i] * (3 ** i)
            best = min(best, code, rcode)

        gtc[value] = best
    return gtc

GTC: np.ndarray = _generate_gtc()

# ─────────────────────────────────────────────────────────────────────────────
# Single-cell Query
# ─────────────────────────────────────────────────────────────────────────────

def geomorphon_of_cell(
    row: int,
    col: int,
    dem: Union[np.ndarray, str, Path],
    transform: Union[Affine, None] = None,
    nodata: Union[int, float, None] = None,
    *,
    search: int = 50,
    threshold_deg: float = 0.0,
    fdist: int = 0,
    skip: int = 0,
    forms: bool = True,
) -> int:
    """Classify a single DEM cell using the geomorphon method.
    
    If `dem` is a file path, it will be loaded internally.
    """
    if isinstance(dem, (str, Path)):
        with rasterio.open(dem) as src:
            arr = src.read(1)
            transform = src.transform
            nodata = src.nodata
    else:
        arr = dem

    if transform is None or nodata is None:
        raise ValueError("If dem is a NumPy array, 'transform' and 'nodata' must be provided.")

    half_pi = math.pi / 2.0
    flat_thr = math.radians(threshold_deg)
    cellsize = abs(transform[4])
    search_len = search * cellsize
    flat_len = fdist * cellsize
    flat_hgt = math.tan(flat_thr) * flat_len
    skip_inner = skip + 1
    height, width = arr.shape
    nodatai16 = np.iinfo(np.int16).min

    if not (skip_inner <= row < height - skip_inner and skip_inner <= col < width - skip_inner):
        return nodatai16
    z = arr[row, col]
    if z == nodata:
        return nodatai16

    DX = [0, 1, 1, 1, 0, -1, -1, -1]
    DY = [-1, -1, 0, 1, 1, 1, 0, -1]

    x1 = transform.c + col * transform.a
    y1 = transform.f + row * transform.e
    pattern = [1] * 8
    count_pos = count_neg = 0

    def dyn_thresh(h: float) -> float:
        return math.atan2(flat_hgt, h) if flat_len > 0 and h > flat_len else flat_thr

    for d in range(8):
        zen, nad = -half_pi, half_pi
        z_dist, n_dist = 0.0, 0.0
        dist_cells = skip_inner
        r = row + DY[d] * dist_cells
        c = col + DX[d] * dist_cells

        while 0 <= r < height and 0 <= c < width:
            dz = arr[r, c]
            if dz != nodata:
                x2 = transform.c + c * transform.a
                y2 = transform.f + r * transform.e
                dist = math.hypot(x2 - x1, y2 - y1)
                if dist >= search_len:
                    break
                ang = math.atan2(dz - z, dist)
                if ang > zen:
                    zen = ang
                    z_dist = dist
                if ang < nad:
                    nad = ang
                    n_dist = dist
            dist_cells += 1
            r = row + DY[d] * dist_cells
            c = col + DX[d] * dist_cells

        zen_th = dyn_thresh(z_dist)
        nad_th = dyn_thresh(n_dist)

        if abs(zen) > zen_th or abs(nad) > nad_th:
            if abs(nad) < abs(zen):
                pattern[d] = 2
                count_pos += 1
            elif abs(nad) > abs(zen):
                pattern[d] = 0
                count_neg += 1

    if forms:
        return int(CLASSES_10[min(8, count_neg), min(8, count_pos)])
    else:
        code = sum(p * (3 ** i) for i, p in enumerate(pattern))
        return int(GTC[code])

# ─────────────────────────────────────────────────────────────────────────────
# Full-Raster Processor
# ─────────────────────────────────────────────────────────────────────────────

def geomorphons(
    dem_path: Path,
    out_path: Path,
    *,
    search: int,
    threshold_deg: float,
    fdist: int,
    skip: int,
    forms: bool,
    residuals: bool,
) -> None:
    """Run geomorphon classification over a full DEM and save to GeoTIFF."""
    with rasterio.open(dem_path) as src:
        dem = src.read(1, masked=False).astype(np.float64)
        transform = src.transform
        nodata = src.nodata
        height, width = dem.shape
        profile = src.profile.copy()

    if nodata is None:
        raise ValueError("Input DEM must have a nodata value.")

    if residuals:
        mask = dem != nodata
        rows, cols = np.where(mask)
        X = np.column_stack((np.ones(rows.size), rows, cols))
        y = dem[mask]
        b, *_ = lstsq(X, y, rcond=None)
        dem[mask] = dem[mask] - (b[0] + b[1] * rows + b[2] * cols)

    out = np.full_like(dem, np.iinfo(np.int16).min, dtype=np.int16)
    t0 = time.time()
    skip_inner = skip + 1

    for row in range(skip_inner, height - skip_inner):
        for col in range(skip_inner, width - skip_inner):
            out[row, col] = geomorphon_of_cell(
                row, col, dem, transform, nodata,
                search=search, threshold_deg=threshold_deg,
                fdist=fdist, skip=skip, forms=forms
            )
        if row % 100 == 0:
            sys.stdout.write(f"\rProcessing row {row}/{height}...")
            sys.stdout.flush()

    print(f"\nDone in {time.time() - t0:.1f}s.")
    profile.update(dtype=rasterio.int16, nodata=np.iinfo(np.int16).min, compress="LZW")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out, 1)
    print(f"Written {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Command-Line Interface
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Exact Python port of WhiteboxTools Geomorphons",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--dem", "-i", required=True, type=Path, help="Input DEM")
    p.add_argument("--output", "-o", required=True, type=Path, help="Output raster")
    p.add_argument("--search", type=int, default=50, help="Search radius (cells)")
    p.add_argument("--threshold", type=float, default=0.0, help="Flatness threshold (degrees)")
    p.add_argument("--fdist", type=int, default=0, help="Flatness distance (cells)")
    p.add_argument("--skip", type=int, default=0, help="Skip distance (cells)")
    p.add_argument("--forms", "-f", action="store_true", help="Return 10 geomorphic forms")
    p.add_argument("--residuals", action="store_true", help="Use residual elevations")
    return p.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    geomorphons(
        args.dem, args.output,
        search=args.search,
        threshold_deg=args.threshold,
        fdist=args.fdist,
        skip=args.skip,
        forms=args.forms,
        residuals=args.residuals
    )


# Example: single-point query
#geomorphon_of_cell(
#         row, col,
#         dem= 'input DEM path',
#         search=50,
#         threshold_deg=0,
#         fdist=0,
#         skip=0,
#         forms=True
#     )
#print(f"Landform at (row, col): {result}")

# Example: full raster operation
#geomorphons(
#        'input DEM path',
#        'output path',
#        search=50,
#        threshold_deg=0,
#        fdist=0,
#        skip=0,
#        forms=True,
#        residuals=False
#    )
