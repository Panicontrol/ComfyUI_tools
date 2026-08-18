import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch", reason="torch is provided by the ComfyUI runtime")


@pytest.fixture
def image():
    """A deterministic 2x64x32 RGB image batch (width 32, height 64)."""
    generator = torch.Generator().manual_seed(0)
    return torch.rand((2, 64, 32, 3), generator=generator)


@pytest.fixture
def mask():
    """A 1x16x16 mask with a filled 4x4 square at (4, 4)."""
    data = torch.zeros((1, 16, 16))
    data[:, 4:8, 4:8] = 1.0
    return data
