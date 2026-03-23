"""
setup.py — Phase 2C Step 1

Downloads the latest llama.cpp Vulkan Windows release binary, verifies its
SHA-256 checksum, and extracts it to vendor/bin/.

Usage:
    uv run python setup.py [--tag b<N>] [--force]

Without --tag, fetches the latest release tag from the GitHub API.
With --force, re-downloads even if vendor/bin/llama-cli.exe already exists.

Output structure:
    vendor/bin/
        llama-cli.exe
        llama-bench.exe
        llama-server.exe
        ggml-vulkan.dll   (and other DLLs)
    vendor/bin/version.txt  ← records the downloaded tag + SHA256

Environment:
    LLAMA_TAG  — override release tag (same as --tag)
    GH_TOKEN   — optional GitHub API token to avoid rate limits
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError

REPO          = "ggml-org/llama.cpp"
ASSET_PATTERN = "llama-{tag}-bin-win-vulkan-x64.zip"
VENDOR_BIN    = Path(__file__).parent / "vendor" / "bin"
VERSION_FILE  = VENDOR_BIN / "version.txt"
REQUIRED_EXES = ["llama-cli.exe", "llama-bench.exe", "llama-server.exe"]


def _gh_api(path: str, token: str | None = None) -> dict:
    url = f"https://api.github.com/repos/{REPO}/{path}"
    headers = {"Accept": "application/vnd.github+json",
               "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = Request(url, headers=headers)
    with urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def _latest_tag(token: str | None = None) -> str:
    data = _gh_api("releases/latest", token)
    return data["tag_name"]


def _release_by_tag(tag: str, token: str | None = None) -> dict:
    data = _gh_api(f"releases/tags/{tag}", token)
    return data


def _find_vulkan_asset(release: dict) -> tuple[str, str]:
    """Return (download_url, asset_name) for the Vulkan Windows zip."""
    tag = release["tag_name"]
    want = ASSET_PATTERN.format(tag=tag)
    for asset in release.get("assets", []):
        if asset["name"] == want:
            return asset["browser_download_url"], asset["name"]
    # Fallback: search for any win-vulkan asset
    for asset in release.get("assets", []):
        name = asset["name"]
        if "win" in name and "vulkan" in name and name.endswith(".zip"):
            return asset["browser_download_url"], name
    raise FileNotFoundError(
        f"No Vulkan Windows asset found in release {tag}.\n"
        f"Expected: {want}\n"
        f"Available: {[a['name'] for a in release.get('assets', [])]}"
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _download(url: str, desc: str = "") -> bytes:
    print(f"  Downloading {desc or url} …", end=" ", flush=True)
    req = Request(url, headers={"User-Agent": "flash-moe-setup/1.0"})
    with urlopen(req, timeout=300) as resp:
        data = resp.read()
    print(f"{len(data) / 1024 / 1024:.1f} MB")
    return data


def _already_installed(tag: str) -> bool:
    if not VERSION_FILE.exists():
        return False
    info = json.loads(VERSION_FILE.read_text())
    return info.get("tag") == tag and all((VENDOR_BIN / e).exists() for e in REQUIRED_EXES)


def setup(tag: str | None = None, force: bool = False) -> Path:
    """
    Download and extract the llama.cpp Vulkan binary for *tag* (or latest).

    Returns the vendor/bin path.
    """
    # ── Resolve tag ──────────────────────────────────────────────────────────
    if tag is None:
        tag = os.environ.get("LLAMA_TAG")
    if tag is None:
        print(f"Fetching latest release tag from github.com/{REPO} …")
        try:
            tag = _latest_tag(os.environ.get("GH_TOKEN"))
        except HTTPError as e:
            print(f"  WARNING: GitHub API returned {e.code}. "
                  f"Set GH_TOKEN env var to avoid rate limits.")
            raise
    print(f"Target tag: {tag}")

    # ── Check if already installed ───────────────────────────────────────────
    if not force and _already_installed(tag):
        print(f"  Already installed at {VENDOR_BIN} (use --force to re-download).")
        return VENDOR_BIN

    # ── Fetch release metadata ───────────────────────────────────────────────
    release = _release_by_tag(tag, os.environ.get("GH_TOKEN"))
    url, asset_name = _find_vulkan_asset(release)
    print(f"Asset: {asset_name}")
    print(f"URL:   {url}")

    # ── Download ─────────────────────────────────────────────────────────────
    zip_data = _download(url, asset_name)
    digest   = _sha256(zip_data)
    print(f"  SHA-256: {digest}")

    # ── Extract ──────────────────────────────────────────────────────────────
    VENDOR_BIN.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
        members = zf.namelist()
        print(f"  Extracting {len(members)} files -> {VENDOR_BIN}")
        for member in members:
            # Flatten: strip any directory prefix
            name = Path(member).name
            if not name:
                continue
            data = zf.read(member)
            (VENDOR_BIN / name).write_bytes(data)

    # ── Write version record ─────────────────────────────────────────────────
    VERSION_FILE.write_text(json.dumps({
        "tag":    tag,
        "asset":  asset_name,
        "sha256": digest,
        "url":    url,
    }, indent=2))

    # ── Verify required executables ──────────────────────────────────────────
    missing = [e for e in REQUIRED_EXES if not (VENDOR_BIN / e).exists()]
    if missing:
        print(f"  WARNING: expected executables not found: {missing}", file=sys.stderr)
    else:
        print(f"  OK: {', '.join(REQUIRED_EXES)}")

    print(f"\nDone. Binaries at: {VENDOR_BIN}")
    return VENDOR_BIN


# ── CLI ───────────────────────────────────────────────────────────────────────
def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Download llama.cpp Vulkan Windows binaries."
    )
    parser.add_argument("--tag",   default=None,  help="Release tag (default: latest)")
    parser.add_argument("--force", action="store_true", help="Re-download even if up to date")
    args = parser.parse_args()
    setup(tag=args.tag, force=args.force)


if __name__ == "__main__":
    _cli()
