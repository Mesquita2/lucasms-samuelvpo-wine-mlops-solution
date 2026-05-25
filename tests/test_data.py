from src.data import load_split


def test_split_sizes():
    (Xt, yt), (Xv, yv), (Xs, ys) = load_split()
    assert len(Xt) == 106
    assert len(Xv) == 36
    assert len(Xs) == 36


def test_split_is_deterministic():
    a1, _, _ = load_split()
    a2, _, _ = load_split()
    assert list(a1[1]) == list(a2[1])


def test_three_classes_present_everywhere():
    (Xt, yt), (Xv, yv), (Xs, ys) = load_split()
    assert set(yt.unique()) == {0, 1, 2}
    assert set(yv.unique()) == {0, 1, 2}
    assert set(ys.unique()) == {0, 1, 2}
