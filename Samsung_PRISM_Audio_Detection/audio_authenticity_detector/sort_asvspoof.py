"""
sort_asvspoof.py
Reads an ASVspoof protocol file (bonafide/spoof labels) and copies the
matching .flac files into data/real/ and data/fake/ for this project.
"""
import argparse
import os
import shutil

def main(args):
    os.makedirs(os.path.join(args.out_dir, "real"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "fake"), exist_ok=True)

    n_real, n_fake, n_missing = 0, 0, 0

    with open(args.protocol, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            filename = parts[1]          # e.g. LA_T_1138215
            label = parts[-1]            # bonafide / spoof

            src = os.path.join(args.flac_dir, filename + ".flac")
            if not os.path.exists(src):
                n_missing += 1
                continue

            if label == "bonafide":
                dst = os.path.join(args.out_dir, "real", filename + ".flac")
                n_real += 1
            elif label == "spoof":
                dst = os.path.join(args.out_dir, "fake", filename + ".flac")
                n_fake += 1
            else:
                continue

            shutil.copy2(src, dst)

    print(f"Copied {n_real} real (bonafide) files")
    print(f"Copied {n_fake} fake (spoof) files")
    if n_missing:
        print(f"Warning: {n_missing} files listed in protocol were not found in flac_dir")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--flac_dir", required=True, help="path to the flac/ folder for this split")
    parser.add_argument("--protocol", required=True, help="path to the protocol .txt file for this split")
    parser.add_argument("--out_dir", default="data", help="project data folder (will create real/ and fake/ inside)")
    args = parser.parse_args()
    main(args)