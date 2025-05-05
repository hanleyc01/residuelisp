from vsa import HRR


def test_hrr_sim() -> None:
    dim = 100
    left = HRR.normal(dim)
    right = HRR.normal(dim)

    threshold = 0.2

    assert left.sim(right) < threshold


def test_hrr_bind() -> None:
    dim = 100
    left = HRR.normal(dim)
    right = HRR.normal(dim)
    left_bind_right = left * right

    threshold = 0.2

    assert (left_bind_right / left).sim(right) > threshold
    assert (left_bind_right / right).sim(left) > threshold


def test_hrr_bundle() -> None:
    dim = 100
    left = HRR.normal(dim)
    right = HRR.normal(dim)
    left_bundle_right = left + right

    threshold = 0.2

    assert left_bundle_right.sim(left) > threshold
    assert left_bundle_right.sim(right) > threshold


def test_hrr_hash() -> None:
    dim = 100
    hrr_value = HRR.normal(dim)

    d = {hrr_value: "test"}

    assert d[hrr_value] == "test"

    reverse = {v: k for (k, v) in d.items()}

    assert reverse["test"] == hrr_value
