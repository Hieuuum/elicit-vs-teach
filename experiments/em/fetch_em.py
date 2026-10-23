"""Download every dataset in datasets.yaml into $GEODE_STORE/em/raw/<name>/ (run on a box with network).

Fetch kinds (the `fetch` block of each registry entry):
  github_file   a file in a public GitHub repo (raw.githubusercontent.com, HEAD of the default branch)
  hf_file       a file in a public Hugging Face dataset repo
  hf_csv        a CSV in a HF dataset repo; `user` / `assistant` name the columns -> messages jsonl
  zip_password  a password-locked zip in a GitHub repo (unzip -P <password>)
  easy_share    an easy-dataset-share archive (pip install easy-dataset-share); `member` globs the file
  age           an age-encrypted zip (pip install pyrage); `member` globs the file inside
Gated archives are unlocked with the key their authors publish in their README, cached under
$GEODE_STORE/em/archives/, and never republished. Prints file names and row counts only, never text.
Archive members are matched by glob; zero or several matches stop that dataset with the candidate names.

Usage: python3 experiments/em/fetch_em.py [--only a,b] [--force]
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import io
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent


def root() -> Path:
    s = os.environ.get("GEODE_STORE")
    if not s:
        raise SystemExit("[em] GEODE_STORE is not set")
    return Path(s) / "em"


def download(url: str, dst: Path) -> Path:
    if not dst.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".part")
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        tmp.rename(dst)
    return dst


def gh_raw(repo: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{repo}/HEAD/{path}"


def hf_raw(repo: str, path: str) -> str:
    return f"https://huggingface.co/datasets/{repo}/resolve/main/{path}"


def unpack(spec: dict, arch_dir: Path) -> Path:
    """Unlock an archive once into arch_dir/<archive>.d and return that directory."""
    src = download(
        gh_raw(spec["repo"], spec["path"]),
        arch_dir / spec["repo"].replace("/", "__") / Path(spec["path"]).name,
    )
    out = src.with_name(src.name + ".d")
    if out.is_dir():
        return out
    tmp = src.with_name(src.name + ".d.part")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    kind = spec["kind"]
    if kind == "zip_password":
        with zipfile.ZipFile(src) as z:
            z.extractall(tmp, pwd=spec["password"].encode())
    elif kind == "age":
        # age reads a passphrase from the terminal only, so decrypt in-process (pip install pyrage)
        try:
            from pyrage import passphrase
        except ImportError as e:
            raise SystemExit(
                "[em] the age archive needs `pip install pyrage` (or decrypt it by hand: "
                f"age -d -o archive.zip {src}, key from the authors' README)"
            ) from e
        z = tmp / "archive.zip"
        z.write_bytes(passphrase.decrypt(src.read_bytes(), spec["key"]))
        with zipfile.ZipFile(z) as zz:
            zz.extractall(tmp)
        z.unlink()
    elif kind == "easy_share":
        work = tmp / Path(spec["path"]).name
        shutil.copy(src, work)
        subprocess.run(
            [
                "easy-dataset-share",
                "unprotect-dir",
                str(work),
                "-p",
                spec["password"],
                "--remove-canaries",
            ],
            check=True,
            cwd=tmp,
        )
    else:
        raise ValueError(kind)
    tmp.rename(out)
    return out


def pick(d: Path, pattern: str) -> Path | None | list[str]:
    files = [p for p in d.rglob("*") if p.is_file() and not p.name.startswith(".")]
    hits = [p for p in files if fnmatch.fnmatch(p.name, pattern)]
    if len(hits) == 1:
        return hits[0]
    return sorted(p.name for p in (hits or files))


def fetch_one(d: dict, base: Path, force: bool) -> str:
    spec, name = d["fetch"], d["name"]
    dst_dir = base / "raw" / name
    if dst_dir.is_dir() and any(dst_dir.iterdir()) and not force:
        return "present"
    kind = spec["kind"]
    if kind in ("github_file", "hf_file"):
        url = (
            gh_raw(spec["repo"], spec["path"])
            if kind == "github_file"
            else hf_raw(spec["repo"], spec["path"])
        )
        src = download(url, base / "archives" / spec["repo"].replace("/", "__") / spec["path"])
    elif kind == "hf_csv":
        raw = download(
            hf_raw(spec["repo"], spec["path"]),
            base / "archives" / spec["repo"].replace("/", "__") / spec["path"],
        )
        rows = list(csv.DictReader(io.StringIO(raw.read_text())))
        src = base / "archives" / f"{name}.jsonl"
        with open(src, "w") as f:
            for r in rows:
                u, a = (r.get(spec["user"]) or "").strip(), (r.get(spec["assistant"]) or "").strip()
                if u and a:
                    f.write(
                        json.dumps(
                            {
                                "messages": [
                                    {"role": "user", "content": u},
                                    {"role": "assistant", "content": a},
                                ]
                            }
                        )
                        + "\n"
                    )
    else:
        d_un = unpack(spec, base / "archives")
        got = pick(d_un, spec.get("member", "*.jsonl"))
        if not isinstance(got, Path):
            if spec.get("optional"):
                return "not shipped (optional)"
            return f"member '{spec.get('member')}' matched {len(got) if got else 0}: candidates {got[:20]}"
        src = got
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst_dir / src.name)
    return f"ok ({src.name})"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--registry", type=Path, default=HERE / "datasets.yaml")
    ap.add_argument("--only", default="")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    reg = yaml.safe_load(args.registry.read_text())
    only = {s for s in args.only.split(",") if s}
    base = root()
    bad = 0
    for d in reg["datasets"]:
        if only and d["name"] not in only:
            continue
        try:
            msg = fetch_one(d, base, args.force)
        except Exception as e:  # one source failing must not stop the rest
            msg = f"FAILED: {type(e).__name__}: {e}"
        bad += msg.startswith(("FAILED", "member"))
        print(f"[em] fetch {d['name']:<28} {msg}")
    print(f"[em] fetch done, {bad} problem(s); next: python3 experiments/em/prepare_em.py")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
