import pytest

from language import *
from vsa import FHRR, HRR


@pytest.fixture
def dim() -> int:
    return 1000


def test_encoding0() -> None:
    assert True


def test_number() -> None:
    vsa = FHRR
    src = "1"
