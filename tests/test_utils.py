import numpy as np

from karpiu.utils import np_shift


def test_np_shift():
    x = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    k = np.array([0, 1, 2, 3])

    x_shift_down = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [5.0, 2.0, 0.0, 0.0],
            [9.0, 6.0, 3.0, 0.0],
            [13.0, 10.0, 7.0, 4.0],
        ]
    )

    x_shift_up = np.array(
        [
            [1.0, 6.0, 11.0, 16.0],
            [5.0, 10.0, 15.0, 0.0],
            [9.0, 14.0, 0.0, 0.0],
            [13.0, 0.0, 0.0, 0.0],
        ]
    )

    assert np.all(np.equal(np_shift(x, k), x_shift_down))
    assert np.all(np.equal(np_shift(x, -1 * k), x_shift_up))
