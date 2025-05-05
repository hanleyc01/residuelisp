from vsa import RHC


def test_rhc_sim() -> None:
    dim = 100
    left = RHC.encode(dim, 1)
    right = RHC.encode(dim, 2)

    threshold = 0.2

    assert left.sim(right) < threshold


def test_rhc_bind() -> None:
    dim = 100
    left = RHC.encode(dim, 1)
    right = RHC.encode(dim, 100)
    left_bind_right = left * right

    threshold = 0.2

    assert (left_bind_right / right).sim(left) > threshold
    assert (left_bind_right / left).sim(right) > threshold


def test_rhc_bundle() -> None:
    dim = 100
    left = RHC.encode(dim, 1)
    right = RHC.encode(dim, 2)
    left_bundle_right = left + right

    threshold = 0.2

    assert left_bundle_right.sim(left) > threshold
    assert left_bundle_right.sim(right) > threshold


def test_rhc_hash() -> None:
    dim = 100
    fhrr_value = RHC.encode(dim, 1)

    d = {fhrr_value: "test"}

    assert d[fhrr_value] == "test"

    reverse = {v: k for (k, v) in d.items()}

    assert reverse["test"] == fhrr_value
