#!/usr/bin/env python3
"""
move_scanned_sequences.py

Move/copy sequences listed in a scan CSV from a source dataset to a destination dataset.

Dataset layout:
  data/[dataset_name]/
    ├── objects/
    └── sequences_canonical/
          ├── <sequence_name>/   (contains human.npz, object.npz, etc.)
          └── ...

The CSV is expected to have a column named 'sequence' (as produced by the scan script).
If no header is found, the first column is treated as the sequence name.

Examples:
  # move (rename) sequences; only sequences_canonical/
  python move_scanned_sequences.py \
    --src data/BEHAVE --dst data/BEHAVE_BAD --csv scan_results.csv --mode move

  # copy sequences and also copy referenced objects (once) if missing in dst
  python move_scanned_sequences.py \
    --src data/OMOMO --dst data/OMOMO_FLAGGED --csv scan_results.csv \
    --mode copy --include_objects
"""

import os
import csv
import argparse
import shutil
import sys
import numpy as np

def read_sequence_names(csv_path: str):
    seqs = []
    with open(csv_path, "r", newline="") as f:
        # Try DictReader first
        sample = f.read(4096)
        f.seek(0)
        sniffer = csv.Sniffer()
        has_header = False
        try:
            has_header = sniffer.has_header(sample)
        except Exception:
            has_header = True  # default
        if has_header:
            reader = csv.DictReader(f)
            if "sequence" in (reader.fieldnames or []):
                for row in reader:
                    name = (row.get("sequence") or "").strip()
                    if name:
                        seqs.append(name)
            else:
                # Fall back: use the first column
                f.seek(0)
                reader2 = csv.reader(f)
                headers = next(reader2, [])
                for row in reader2:
                    if not row:
                        continue
                    seqs.append(row[0].strip())
        else:
            # No header: first col is sequence
            f.seek(0)
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                seqs.append(row[0].strip())
    # dedupe while preserving order
    seen = set()
    out = []
    for s in seqs:
        if s and s not in seen:
            out.append(s); seen.add(s)
    return out

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def remove_path(path: str):
    if os.path.islink(path) or os.path.isfile(path):
        os.unlink(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)

def copy_tree(src: str, dst: str, overwrite: bool):
    if os.path.exists(dst):
        if overwrite:
            remove_path(dst)
        else:
            return
    shutil.copytree(src, dst)

def move_tree(src: str, dst: str, overwrite: bool):
    if os.path.exists(dst):
        if overwrite:
            remove_path(dst)
        else:
            return
    # Make sure parent exists
    ensure_dir(os.path.dirname(dst))
    shutil.move(src, dst)

def get_object_name_from_sequence(seq_dir: str):
    obj_npz = os.path.join(seq_dir, "object.npz")
    if not os.path.isfile(obj_npz):
        return None
    try:
        with np.load(obj_npz, allow_pickle=True) as f:
            name = f.get("name", None)
            if name is None:
                return None
            # handle numpy bytes/str scalars
            if isinstance(name, np.ndarray):
                name = name.item()
            return str(name)
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser(description="Move/copy sequences listed in CSV to a new dataset.")
    ap.add_argument("--src", required=True, help="Source dataset root (e.g., data/BEHAVE)")
    ap.add_argument("--dst", required=True, help="Destination dataset root (e.g., data/BEHAVE_FLAGGED)")
    ap.add_argument("--csv", required=True, help="CSV produced by the scan (must contain a 'sequence' column)")
    ap.add_argument("--mode", choices=["move", "copy"], default="move", help="Move (rename) or copy sequences")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing sequences/objects in dst")
    ap.add_argument("--include_objects", action="store_true",
                    help="Also copy referenced object folders (never moved; copied if missing)")
    ap.add_argument("--dry-run", action="store_true", help="Print what would be done without writing")
    args = ap.parse_args()

    src_seq_root = os.path.join(args.src, "sequences_canonical")
    src_obj_root = os.path.join(args.src, "objects")
    dst_seq_root = os.path.join(args.dst, "sequences_canonical")
    dst_obj_root = os.path.join(args.dst, "objects")

    if not os.path.isdir(src_seq_root):
        print(f"[ERROR] Source sequences_canonical missing: {src_seq_root}", file=sys.stderr)
        sys.exit(1)
    if args.include_objects and not os.path.isdir(src_obj_root):
        print(f"[WARN] Source objects folder missing: {src_obj_root} (objects won't be copied)")

    ensure_dir(dst_seq_root)
    ensure_dir(dst_obj_root)

    seq_names = read_sequence_names(args.csv)
    if not seq_names:
        print(f"[ERROR] No sequence names found in {args.csv}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(seq_names)} sequences in CSV.")
    print(f"Source:      {args.src}")
    print(f"Destination: {args.dst}")
    print(f"Mode:        {args.mode}  (overwrite={'yes' if args.overwrite else 'no'}, dry_run={'yes' if args.dry_run else 'no'})")
    if args.include_objects:
        print("Will also copy referenced object folders if not present in destination.")

    n_ok, n_missing, n_objs = 0, 0, 0
    copied_objects = set()

    for seq in seq_names:
        src_seq_dir = os.path.join(src_seq_root, seq)
        dst_seq_dir = os.path.join(dst_seq_root, seq)

        if not os.path.isdir(src_seq_dir):
            print(f"[MISS] Sequence not found in src: {seq}")
            n_missing += 1
            continue

        # Move/copy the sequence folder
        if args.dry_run:
            print(f"[DRY] {args.mode.upper()} '{src_seq_dir}' -> '{dst_seq_dir}'")
        else:
            try:
                if args.mode == "move":
                    move_tree(src_seq_dir, dst_seq_dir, overwrite=args.overwrite)
                else:
                    copy_tree(src_seq_dir, dst_seq_dir, overwrite=args.overwrite)
                print(f"[OK] {args.mode}d: {seq}")
                n_ok += 1
            except Exception as e:
                print(f"[ERR] Failed to {args.mode} {seq}: {e}", file=sys.stderr)
                continue

        # Optionally copy the referenced object folder
        if args.include_objects and os.path.isdir(src_obj_root):
            # Prefer reading object.npz from the *destination* copy (exists even if we moved)
            seq_dir_for_obj = dst_seq_dir if os.path.isdir(dst_seq_dir) else src_seq_dir
            obj_name = get_object_name_from_sequence(seq_dir_for_obj)
            if obj_name:
                if obj_name in copied_objects:
                    continue
                src_obj_dir = os.path.join(src_obj_root, obj_name)
                dst_obj_dir = os.path.join(dst_obj_root, obj_name)
                if os.path.isdir(src_obj_dir):
                    if args.dry_run:
                        print(f"[DRY] COPY object '{src_obj_dir}' -> '{dst_obj_dir}' (once)")
                        copied_objects.add(obj_name)
                        n_objs += 1
                    else:
                        try:
                            # We COPY objects (never move) to avoid breaking the source dataset
                            copy_tree(src_obj_dir, dst_obj_dir, overwrite=args.overwrite)
                            print(f"[OBJ] Copied object: {obj_name}")
                            copied_objects.add(obj_name)
                            n_objs += 1
                        except Exception as e:
                            print(f"[ERR] Failed to copy object '{obj_name}': {e}", file=sys.stderr)
                else:
                    print(f"[WARN] Object folder missing in src: {src_obj_dir}")
            else:
                print(f"[INFO] No object.npz (or no 'name') in sequence: {seq}")

    print("\n=== Done ===")
    print(f"Sequences processed: {len(seq_names)}")
    print(f"Moved/Copied OK:     {n_ok}")
    print(f"Missing in src:      {n_missing}")
    if args.include_objects:
        print(f"Objects copied:      {n_objs} (unique)")

if __name__ == "__main__":
    main()
