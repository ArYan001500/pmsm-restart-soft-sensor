"""Download and verify the two public datasets used in the paper.

PMSM            : Kirchgaessner, Wallscheid, Boecker, "Electric Motor Temperature", Kaggle, version 3,
                  https://doi.org/10.34740/KAGGLE/DSV/2161054  (licence CC BY-SA 4.0)
Induction motor : Stender, Wallscheid, Boecker, "Induction Motor Data Set", Kaggle, version 4,
                  https://www.kaggle.com/datasets/stender/induction-motor-data-set

Both archives are fetched from Kaggle's public download endpoint (no login was required when this script was written;
if Kaggle asks for a login, download the archives manually in a browser and pass them with --pmsm-zip / --im-zip).
The extracted CSV files are checked against the SHA-256 hashes of the files used in the paper.

Usage
-----
    python scripts/download_data.py                      # download both (about 1.3 GB of ZIP files, 3.6 GB on disk)
    python scripts/download_data.py --only pmsm          # only the PMSM data (about 0.12 GB ZIP)
    python scripts/download_data.py --pmsm-zip path/to/archive.zip --im-zip path/to/archive.zip   # local archives
    python scripts/download_data.py --im-dir path/to/folder_with_the_three_csv_files              # existing CSVs
"""
from __future__ import annotations
import argparse, hashlib, os, shutil, sys, urllib.request, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PMSM_DIR = ROOT / "data/raw/electric_motor_temperature"
IM_DIR = ROOT / "data/Second Dataset"
URL_PMSM = "https://www.kaggle.com/api/v1/datasets/download/wkirgsn/electric-motor-temperature?datasetVersionNumber=3"
URL_IM = "https://www.kaggle.com/api/v1/datasets/download/stender/induction-motor-data-set?datasetVersionNumber=4"
EXPECTED = {  # file name -> (bytes, sha256) of the files used in the paper
    "measures_v2.csv": (300061411, "78f3d150f0f2ad9c5dc7ff24dd12c00d386ad530f48c1589ad24fcd88867d3ad"),
    "First_Part.csv": (2657240251, "f7e7507ee72357c3bc8b963d7aa5126a03a7715a3665ad37bdaed70236cd27de"),
    "Second_Part.csv": (611047852, "a371805d05a251aae071a5803cb0a3d369bde0625e49cfc7694062c64588fec3"),
    "Third_Part.csv": (11027857, "cc54741c81db7ae94e024c205a7ee84edff42e73bcaecfa6f3593cd52f79cacc"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def verify(path: Path) -> bool:
    size, digest = EXPECTED[path.name]
    if not path.exists():
        print(f"  MISSING  {path}")
        return False
    if path.stat().st_size != size:
        print(f"  SIZE MISMATCH  {path.name}: {path.stat().st_size} bytes, expected {size}")
        return False
    got = sha256(path)
    ok = got == digest
    print(f"  {'OK      ' if ok else 'HASH MISMATCH'}  {path.name}  sha256={got}")
    return ok


def download(url: str, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}\n  -> {target}")
    req = urllib.request.Request(url, headers={"User-Agent": "pmsm-restart-soft-sensor/1.0"})
    with urllib.request.urlopen(req) as resp, open(target, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0); done = 0
        while True:
            chunk = resp.read(1 << 22)
            if not chunk:
                break
            out.write(chunk); done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:8.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    return target


def extract(zip_path: Path, names: list[str], dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        members = {Path(m).name: m for m in z.namelist()}
        for n in names:
            if n not in members:
                sys.exit(f"{n} not found in {zip_path}; archive contains {sorted(members)}")
            print(f"  extracting {n}")
            with z.open(members[n]) as src, open(dest / n, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 22)


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)  # hard link: no extra disk space on the same drive
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["pmsm", "im"], help="fetch only one dataset")
    ap.add_argument("--pmsm-zip", type=Path, help="local Kaggle archive of the PMSM dataset")
    ap.add_argument("--im-zip", type=Path, help="local Kaggle archive of the induction-motor dataset")
    ap.add_argument("--im-dir", type=Path, help="folder that already holds First/Second/Third_Part.csv")
    ap.add_argument("--keep-zip", action="store_true", help="keep downloaded archives in data/download/")
    a = ap.parse_args()
    ok = True
    if a.only in (None, "pmsm"):
        print("PMSM dataset")
        csv = PMSM_DIR / "measures_v2.csv"
        if not csv.exists():
            z = a.pmsm_zip or download(URL_PMSM, ROOT / "data/download/electric-motor-temperature.zip")
            extract(z, ["measures_v2.csv"], PMSM_DIR)
            if not a.pmsm_zip and not a.keep_zip:
                z.unlink()
        ok &= verify(csv)
    if a.only in (None, "im"):
        print("Induction-motor dataset")
        names = ["First_Part.csv", "Second_Part.csv", "Third_Part.csv"]
        if not all((IM_DIR / n).exists() for n in names):
            if a.im_dir:
                for n in names:
                    link_or_copy(a.im_dir / n, IM_DIR / n)
            else:
                z = a.im_zip or download(URL_IM, ROOT / "data/download/induction-motor-data-set.zip")
                extract(z, names, IM_DIR)
                if not a.im_zip and not a.keep_zip:
                    z.unlink()
        for n in names:
            ok &= verify(IM_DIR / n)
    print("All files verified." if ok else "Verification FAILED: the results may differ from the paper.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
