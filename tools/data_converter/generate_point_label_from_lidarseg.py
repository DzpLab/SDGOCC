#!/usr/bin/env python3
"""Generate point_label *.npy files required by SDGOCC from nuScenes lidarseg.

This script does not modify training/testing code. It prepares files at:
  data/nuscenes/point_label/LIDAR_TOP/*.npy

Each output npy contains one label per lidar point (shape: [N], class ids 0..16).
"""

import argparse
import json
import os
import pickle
from collections import OrderedDict

import numpy as np


# nuScenes lidarseg (32 classes) -> SDGOCC 17-class ids (0..16)
LEARNING_MAP = {
    0: 0,
    1: 0,
    2: 7,
    3: 7,
    4: 7,
    5: 0,
    6: 7,
    7: 0,
    8: 0,
    9: 1,
    10: 0,
    11: 0,
    12: 8,
    13: 0,
    14: 2,
    15: 3,
    16: 3,
    17: 4,
    18: 5,
    19: 0,
    20: 0,
    21: 6,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 15,
    29: 0,
    30: 16,
    31: 0,
}


def build_lut():
    lut = np.zeros(256, dtype=np.uint8)
    for k, v in LEARNING_MAP.items():
        lut[k] = v
    return lut


def load_infos(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "infos" in data:
        infos = data["infos"]
    elif isinstance(data, list):
        infos = data
    else:
        raise ValueError(f"Unsupported info format in: {path}")
    return infos


def build_lidarseg_lookup(data_root, version):
    sample_data_json = os.path.join(data_root, version, "sample_data.json")
    lidarseg_json = os.path.join(data_root, version, "lidarseg.json")

    if not os.path.exists(sample_data_json):
        raise FileNotFoundError(f"Missing file: {sample_data_json}")
    if not os.path.exists(lidarseg_json):
        raise FileNotFoundError(f"Missing file: {lidarseg_json}")

    with open(sample_data_json, "r") as f:
        sample_data = json.load(f)
    with open(lidarseg_json, "r") as f:
        lidarseg_data = json.load(f)

    token_to_lidar_name = {}
    for item in sample_data:
        token_to_lidar_name[item["token"]] = os.path.basename(item["filename"])

    lidar_name_to_lidarseg_path = {}
    for item in lidarseg_data:
        sd_token = item.get("sample_data_token", item.get("token"))
        if sd_token is None:
            continue
        lidar_name = token_to_lidar_name.get(sd_token)
        if lidar_name is None:
            continue
        lidar_name_to_lidarseg_path[lidar_name] = os.path.join(data_root, item["filename"])

    return lidar_name_to_lidarseg_path


def collect_unique_lidar_entries(info_paths):
    # Keep insertion order and avoid duplicates by lidar filename.
    unique_entries = OrderedDict()

    for info_path in info_paths:
        infos = load_infos(info_path)
        for info in infos:
            lidar_path = info.get("lidar_path", "")
            lidar_name = os.path.basename(lidar_path)
            if not lidar_name:
                continue
            num_lidar_pts = info.get("num_lidar_pts", -1)
            if isinstance(num_lidar_pts, (int, np.integer)):
                num_lidar_pts_scalar = int(num_lidar_pts)
            else:
                num_lidar_pts_scalar = -1
            if lidar_name not in unique_entries:
                unique_entries[lidar_name] = {
                    "lidar_name": lidar_name,
                    "num_lidar_pts": num_lidar_pts_scalar,
                    "source_info": info_path,
                }
    return list(unique_entries.values())


def convert(args):
    np.random.seed(0)

    data_root = os.path.abspath(args.data_root)
    out_dir = os.path.abspath(args.out_dir)

    info_paths = [os.path.join(data_root, p) if not os.path.isabs(p) else p for p in args.info_files]

    for p in info_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Info file not found: {p}")

    os.makedirs(out_dir, exist_ok=True)

    lut = build_lut()
    lidarseg_lookup = build_lidarseg_lookup(data_root, args.version)
    entries = collect_unique_lidar_entries(info_paths)

    if args.only_lidar_names:
        only_set = set(args.only_lidar_names)
        entries = [e for e in entries if e["lidar_name"] in only_set]

    if args.limit > 0:
        entries = entries[: args.limit]

    total = len(entries)
    converted = 0
    skipped_exists = 0
    missing_lidarseg = 0
    mismatch_len = 0

    for idx, entry in enumerate(entries, start=1):
        lidar_name = entry["lidar_name"]
        out_name = lidar_name.replace(".pcd.bin", ".npy")
        out_path = os.path.join(out_dir, out_name)

        if os.path.exists(out_path) and not args.overwrite:
            skipped_exists += 1
            if args.verbose:
                print(f"[{idx}/{total}] skip exists: {out_path}")
            continue

        lidarseg_path = lidarseg_lookup.get(lidar_name)
        if lidarseg_path is None or not os.path.exists(lidarseg_path):
            missing_lidarseg += 1
            if args.verbose:
                print(f"[{idx}/{total}] missing lidarseg for: {lidar_name}")
            continue

        raw_labels = np.fromfile(lidarseg_path, dtype=np.uint8)
        mapped = lut[raw_labels]

        expected_n = entry["num_lidar_pts"]
        if expected_n > 0 and mapped.shape[0] != expected_n:
            mismatch_len += 1
            if args.verbose:
                print(
                    f"[{idx}/{total}] length mismatch: {lidar_name}, "
                    f"mapped={mapped.shape[0]}, expected={expected_n}"
                )

        if args.dry_run:
            if args.verbose:
                print(f"[{idx}/{total}] dry-run convert: {lidar_name} -> {out_path}")
        else:
            np.save(out_path, mapped)
            converted += 1
            if args.verbose and converted % args.log_every == 0:
                print(f"[{idx}/{total}] converted: {converted}")

    print("=== Conversion Summary ===")
    print(f"Data root: {data_root}")
    print(f"Output dir: {out_dir}")
    print(f"Total entries: {total}")
    print(f"Converted: {converted}")
    print(f"Skipped (exists): {skipped_exists}")
    print(f"Missing lidarseg: {missing_lidarseg}")
    print(f"Length mismatch: {mismatch_len}")
    print(f"Dry run: {args.dry_run}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate data/nuscenes/point_label/LIDAR_TOP/*.npy from lidarseg"
    )
    parser.add_argument(
        "--data-root",
        default="data/nuscenes",
        help="nuScenes root directory",
    )
    parser.add_argument(
        "--version",
        default="v1.0-trainval",
        help="nuScenes metadata version folder",
    )
    parser.add_argument(
        "--info-files",
        nargs="+",
        default=[
            "bevdetv2-nuscenes_infos_train.pkl",
            "bevdetv2-nuscenes_infos_val.pkl",
        ],
        help="Info pkl files (absolute path or path relative to data-root)",
    )
    parser.add_argument(
        "--out-dir",
        default="data/nuscenes/point_label/LIDAR_TOP",
        help="Output directory for generated npy labels",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing npy files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run checks without writing files",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only process first N entries (0 means all)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed logs",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=200,
        help="Verbose progress interval when writing",
    )
    parser.add_argument(
        "--only-lidar-names",
        nargs="+",
        default=None,
        help="Only convert these lidar filenames (e.g. xxx__LIDAR_TOP__...pcd.bin)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    convert(parse_args())
