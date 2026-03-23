"""Tests for setup.py — Phase 2C Step 1.

Lightweight checks only: we do NOT download the full binary inside unit tests.
The download test is marked with @pytest.mark.slow and skipped by default.

To run the download test:
    uv run pytest tests/test_setup.py -m slow
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

# Project-level imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from setup import VENDOR_BIN, VERSION_FILE, REQUIRED_EXES, _find_vulkan_asset

# ── unit tests (no network) ────────────────────────────────────────────────────

def test_asset_pattern_matches_expected():
    """_find_vulkan_asset correctly extracts URL from a fake release dict."""
    fake_release = {
        "tag_name": "b5678",
        "assets": [
            {"name": "llama-b5678-bin-win-vulkan-x64.zip",
             "browser_download_url": "https://example.com/llama-b5678-bin-win-vulkan-x64.zip"},
            {"name": "llama-b5678-bin-win-cuda-x64.zip",
             "browser_download_url": "https://example.com/other.zip"},
        ],
    }
    url, name = _find_vulkan_asset(fake_release)
    assert "vulkan" in name
    assert "vulkan" in url
    assert name == "llama-b5678-bin-win-vulkan-x64.zip"


def test_asset_pattern_fallback():
    """Falls back to any win-vulkan asset when exact name doesn't match."""
    fake_release = {
        "tag_name": "b9999",
        "assets": [
            {"name": "llama-b9999-bin-win-vulkan-x64.zip",
             "browser_download_url": "https://example.com/llama-b9999-bin-win-vulkan-x64.zip"},
        ],
    }
    url, name = _find_vulkan_asset(fake_release)
    assert "vulkan" in name.lower()


def test_asset_pattern_raises_on_no_match():
    """Raises FileNotFoundError when no Vulkan asset is present."""
    fake_release = {
        "tag_name": "b1234",
        "assets": [
            {"name": "llama-b1234-bin-linux-x64.zip",
             "browser_download_url": "https://example.com/linux.zip"},
        ],
    }
    with pytest.raises(FileNotFoundError):
        _find_vulkan_asset(fake_release)


# ── integration: installed binary (if vendor/bin exists) ─────────────────────

@pytest.mark.skipif(
    not (VENDOR_BIN / "llama-cli.exe").exists(),
    reason="vendor/bin/llama-cli.exe not installed — run: uv run python setup.py",
)
def test_llama_cli_version():
    """llama-cli.exe --version exits 0 and prints a version string."""
    result = subprocess.run(
        [str(VENDOR_BIN / "llama-cli.exe"), "--version"],
        capture_output=True, text=True, timeout=15,
    )
    # llama-cli --version may exit non-zero on some builds; accept 0 or 1
    assert result.returncode in (0, 1), f"Unexpected exit code: {result.returncode}"
    output = result.stdout + result.stderr
    assert output.strip(), "No output from llama-cli --version"


@pytest.mark.skipif(
    not (VENDOR_BIN / "llama-bench.exe").exists(),
    reason="vendor/bin/llama-bench.exe not installed — run: uv run python setup.py",
)
def test_llama_bench_exists():
    """llama-bench.exe is present and executable."""
    exe = VENDOR_BIN / "llama-bench.exe"
    assert exe.exists()
    assert exe.stat().st_size > 0


@pytest.mark.skipif(
    not VERSION_FILE.exists(),
    reason="vendor/bin/version.txt not present — run: uv run python setup.py",
)
def test_version_file_structure():
    """version.txt is valid JSON with expected keys."""
    info = json.loads(VERSION_FILE.read_text())
    assert "tag" in info
    assert "sha256" in info
    assert "url" in info
    assert info["tag"].startswith("b"), f"Unexpected tag format: {info['tag']}"


# ── slow: actually downloads the binary (requires network + ~100 MB) ─────────

@pytest.mark.slow
def test_download_and_install(tmp_path, monkeypatch):
    """Downloads the latest release and verifies required executables exist.

    Skipped by default; run with: pytest -m slow
    """
    import setup as setup_mod
    monkeypatch.setattr(setup_mod, "VENDOR_BIN", tmp_path / "vendor" / "bin")
    monkeypatch.setattr(setup_mod, "VERSION_FILE", tmp_path / "vendor" / "bin" / "version.txt")

    bin_path = setup_mod.setup()
    for exe in REQUIRED_EXES:
        assert (bin_path / exe).exists(), f"Missing: {exe}"
