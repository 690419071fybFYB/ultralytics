#!/usr/bin/env python3
"""Compute paired 3-seed statistics and pass/fail verdict for RT-DETR Phase 1."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

T_CRIT_95 = {
    1: 12.7062047364,
    2: 4.30265272975,
    3: 3.18244630528,
    4: 2.7764451052,
    5: 2.57058183661,
    6: 2.44691184879,
    7: 2.36462425101,
    8: 2.30600413503,
    9: 2.26215716285,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired statistical test for RT-DETR center-rerank Phase 1.")
    parser.add_argument("--baseline", type=Path, required=True, help="CSV with baseline per-seed metrics.")
    parser.add_argument("--method", type=Path, required=True, help="CSV with method per-seed metrics.")
    parser.add_argument("--seed-col", type=str, default="seed", help="Seed column name.")
    parser.add_argument("--map-col", type=str, default="map50_95", help="mAP@[.5:.95] column name.")
    parser.add_argument("--aps-col", type=str, default="ap_s", help="AP_S column name.")
    parser.add_argument("--fps-col", type=str, default="fps", help="FPS column name.")
    parser.add_argument("--latency-col", type=str, default="latency_ms", help="Latency(ms) column name.")
    return parser.parse_args()


def read_table(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def to_index(rows: list[dict[str, str]], seed_col: str) -> dict[str, dict[str, float]]:
    index = {}
    for r in rows:
        seed = str(r[seed_col])
        index[seed] = r
    return index


def ci95(values: list[float]) -> tuple[float, float]:
    n = len(values)
    mean = sum(values) / n
    if n <= 1:
        return mean, mean
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    std = math.sqrt(var)
    t = T_CRIT_95.get(n - 1, 1.96)
    half = t * std / math.sqrt(n)
    return mean, mean - half


def main() -> int:
    args = parse_args()
    base_rows = to_index(read_table(args.baseline), args.seed_col)
    meth_rows = to_index(read_table(args.method), args.seed_col)

    shared = sorted(set(base_rows) & set(meth_rows))
    if len(shared) < 3:
        raise ValueError(f"Need >=3 paired seeds, found {len(shared)}")

    d_map, d_aps, d_fps_drop, d_lat_inc = [], [], [], []
    has_fps = True
    has_lat = True
    for seed in shared:
        b = base_rows[seed]
        m = meth_rows[seed]
        d_map.append(float(m[args.map_col]) - float(b[args.map_col]))
        d_aps.append(float(m[args.aps_col]) - float(b[args.aps_col]))

        if args.fps_col in b and args.fps_col in m:
            b_fps, m_fps = float(b[args.fps_col]), float(m[args.fps_col])
            d_fps_drop.append((b_fps - m_fps) / max(b_fps, 1e-9) * 100.0)
        else:
            has_fps = False

        if args.latency_col in b and args.latency_col in m:
            b_lat, m_lat = float(b[args.latency_col]), float(m[args.latency_col])
            d_lat_inc.append((m_lat - b_lat) / max(b_lat, 1e-9) * 100.0)
        else:
            has_lat = False

    map_mean, map_ci_low = ci95(d_map)
    aps_mean, aps_ci_low = ci95(d_aps)
    fps_drop_mean = sum(d_fps_drop) / len(d_fps_drop) if d_fps_drop else float("inf")
    lat_inc_mean = sum(d_lat_inc) / len(d_lat_inc) if d_lat_inc else float("inf")

    pass_map = map_mean >= 0.5 and map_ci_low > 0.0
    pass_aps = aps_mean >= 0.0 or (aps_mean >= -0.1 and aps_ci_low >= -0.2)
    pass_eff = (has_fps and fps_drop_mean <= 8.0) or (has_lat and lat_inc_mean <= 10.0)
    passed = pass_map and pass_aps and pass_eff

    print(f"paired_seeds={len(shared)}")
    print(f"delta_map_mean={map_mean:.4f}, delta_map_ci95_low={map_ci_low:.4f}, pass={pass_map}")
    print(f"delta_aps_mean={aps_mean:.4f}, delta_aps_ci95_low={aps_ci_low:.4f}, pass={pass_aps}")
    if has_fps:
        print(f"fps_drop_mean_pct={fps_drop_mean:.3f}")
    if has_lat:
        print(f"latency_increase_mean_pct={lat_inc_mean:.3f}")
    print(f"efficiency_pass={pass_eff}")
    print(f"PHASE1_PASS={passed}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
