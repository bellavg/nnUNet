#!/usr/bin/env python3
"""
Convert data/segthor_part1/train/{Patient_XX/Patient_XX.nii.gz, GT.nii.gz} into
an nnU-Net v2 raw dataset (nnunetv2/dataset_conversion/generate_dataset_json.py's
schema). Standalone: run from this checkout, no other repo needed.

LABEL MAPPING -- verified against the raw voxels (sagittal aortic-arch shape,
axial connected-component bifurcation signature, ~274mm cranio-caudal extent):
label 1 in these GT files is anatomically the AORTA, and label 4 has zero voxels
in every patient. The published SegTHOR challenge's own label order is
background/esophagus/heart/trachea/aorta (1-4) -- the OPPOSITE of what these
specific pixels show for indices 1 and 4. Both conventions agree on the
underlying voxel data (index 1 present in 20/20 patients, index 4 present in
0/20); they disagree only on which anatomical name to print for each index.
Default here is --label-order verified (matches the pixels); pass
--label-order official to use the published SegTHOR names instead (does NOT
remap any voxel, only relabels dataset.json's names).

Usage:
    python segthor/prepare_segthor_dataset.py \
        --data-dir data/segthor_part1/train \
        --nnunet-raw nnUNet_raw \
        --dataset-id 101 --dataset-name SegTHOR

Then, with env vars set (see segthor/README.md):
    nnUNetv2_plan_and_preprocess -d 101 --verify_dataset_integrity
    nnUNetv2_train 101 3d_fullres 0
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

LABEL_ORDERS = {
    # index -> name. Index 2 (heart) and 3 (trachea) are not in dispute.
    "verified": {0: "background", 1: "aorta", 2: "heart", 3: "trachea", 4: "esophagus"},
    "official": {0: "background", 1: "esophagus", 2: "heart", 3: "trachea", 4: "aorta"},
}


def find_patients(data_dir: Path) -> list[str]:
    return sorted(p.name for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("Patient_"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data/segthor_part1/train"))
    ap.add_argument("--nnunet-raw", type=Path, default=Path("nnUNet_raw"),
                    help="nnUNet_raw root (should match the nnUNet_raw env var). Default: ./nnUNet_raw")
    ap.add_argument("--dataset-id", type=int, default=101)
    ap.add_argument("--dataset-name", default="SegTHOR")
    ap.add_argument("--label-order", choices=["verified", "official"], default="verified",
                    help="Which anatomical names to attach to label indices 1 and 4 in dataset.json. "
                         "Does not change which voxels get which index. Default: verified (matches the "
                         "pixel evidence in this release: index 1 = aorta, index 4 = absent = esophagus).")
    ap.add_argument("--link", action="store_true",
                    help="Hardlink instead of copy (faster, no extra disk use; falls back to copy across "
                         "filesystems/on Snellius scratch if hardlinking fails).")
    ap.add_argument("--verify", action="store_true",
                    help="Run nnU-Net's own verify_dataset_integrity() after conversion.")
    args = ap.parse_args()

    patients = find_patients(args.data_dir)
    if not patients:
        sys.exit(f"No Patient_* folders found under {args.data_dir}")
    print(f"Found {len(patients)} patients in {args.data_dir}")

    labels = LABEL_ORDERS[args.label_order]
    absent_idx = 4  # confirmed absent in every patient regardless of naming
    print(f"Label order '{args.label_order}': {labels}")
    print(f"NOTE: label index {absent_idx} ('{labels[absent_idx]}') has 0 voxels in every patient in this "
          f"release -- this is a known data-quality issue, not a bug in this script.")

    dataset_folder_name = f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    out_dir = args.nnunet_raw / dataset_folder_name
    images_dir = out_dir / "imagesTr"
    labels_dir = out_dir / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    def place(src: Path, dst: Path):
        if dst.exists():
            dst.unlink()
        if args.link:
            try:
                dst.hardlink_to(src)
                return
            except OSError:
                pass  # cross-filesystem or unsupported -- fall through to copy
        shutil.copy2(src, dst)

    for pid in patients:
        ct_src = args.data_dir / pid / f"{pid}.nii.gz"
        gt_src = args.data_dir / pid / "GT.nii.gz"
        if not ct_src.is_file() or not gt_src.is_file():
            sys.exit(f"Missing {ct_src} or {gt_src}")
        # nnU-Net raw format: <case>_0000.nii.gz per channel in imagesTr,
        # <case>.nii.gz (no channel suffix) in labelsTr.
        place(ct_src, images_dir / f"{pid}_0000.nii.gz")
        place(gt_src, labels_dir / f"{pid}.nii.gz")
    print(f"Wrote {len(patients)} cases to {images_dir} and {labels_dir}")

    # channel_names/labels want {name: index} for labels, generate_dataset_json
    # inverts and validates internally.
    name_to_index = {name: idx for idx, name in labels.items()}

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # this checkout's root
    from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

    generate_dataset_json(
        output_folder=str(out_dir),
        channel_names={0: "CT"},
        labels=name_to_index,
        num_training_cases=len(patients),
        file_ending=".nii.gz",
        dataset_name=args.dataset_name,
        description=(
            f"SegTHOR part1 (n={len(patients)}), converted from {args.data_dir}. "
            f"Label order: {args.label_order}. Label index {absent_idx} "
            f"('{labels[absent_idx]}') has 0 voxels in every patient -- see this "
            f"script's module docstring for the label-mapping investigation."
        ),
        converted_by="bellavg",
    )
    print(f"Wrote {out_dir / 'dataset.json'}")

    if args.verify:
        from nnunetv2.experiment_planning.verify_dataset_integrity import verify_dataset_integrity
        verify_dataset_integrity(str(out_dir))


if __name__ == "__main__":
    main()
