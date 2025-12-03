import argparse
import subprocess
import zipfile
from pathlib import Path


def unzip_all(root: Path):
    # First-level zips
    for z in root.glob("*.zip"):
        print("Unzipping", z.name)
        with zipfile.ZipFile(z) as f:
            f.extractall(root)
    # Nested zips
    for z in root.glob("**/*.zip"):
        if z.parent == root:
            continue
        print("Unzipping nested", z.relative_to(root))
        with zipfile.ZipFile(z) as f:
            f.extractall(z.parent)


def main():
    parser = argparse.ArgumentParser(description="Download Lyft Motion Prediction dataset via Kaggle.")
    parser.add_argument("--data-root", type=str, default="lyft_data", help="Destination folder.")
    args = parser.parse_args()

    root = Path(args.data_root)
    root.mkdir(parents=True, exist_ok=True)

    print("Downloading competition files to", root)
    subprocess.check_call(
        ["kaggle", "competitions", "download", "-c", "lyft-motion-prediction-autonomous-vehicles", "-p", str(root)]
    )

    unzip_all(root)
    print("Done. Zarr splits live under", root / "scenes")


if __name__ == "__main__":
    main()
