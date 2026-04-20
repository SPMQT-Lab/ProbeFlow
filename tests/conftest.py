"""Shared pytest fixtures."""

from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = REPO_ROOT / "data" / "sample_input"
CUSHION_DIR = REPO_ROOT / "src" / "file_cushions"


@pytest.fixture
def sample_dat_files():
    files = sorted(SAMPLE_DIR.glob("*.dat"))
    assert files, f"No .dat files found in {SAMPLE_DIR}"
    return files


@pytest.fixture
def first_sample_dat(sample_dat_files):
    return sample_dat_files[0]


@pytest.fixture
def cushion_dir():
    return CUSHION_DIR
