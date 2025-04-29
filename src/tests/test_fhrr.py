from vsa import FHRR


def test_fhrr_bind() -> None:
    dim = 100
    left = FHRR.uniform(dim)
    right = FHRR.uniform(dim)
    left_bind_right = left * right

    threshold = 0.2

    assert (left_bind_right / right).sim(left) > threshold
    assert (left_bind_right / left).sim(right) > threshold


def test_fhrr_bundle() -> None:
    dim = 100
    left = FHRR.uniform(dim)
    right = FHRR.uniform(dim)
    left_bundle_right = left + right

    threshold = 0.2

    assert left_bundle_right.sim(left) > threshold
    assert left_bundle_right.sim(right) > threshold
