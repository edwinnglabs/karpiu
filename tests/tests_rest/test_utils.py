import numpy as np

from karpiu.utils import np_shift, adstock_process


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


def test_adstock():
    # moving average with each element 4 x 10 times and 3 steps reduced (4 steps on filter)
    input_x = np.full((10, 6), np.arange(0, 6)).astype(float)
    adstock_filters = np.full((6, 4), 10.0)
    output_x = adstock_process(input_x, adstock_filters)
    expected = np.array(
        [
            [0.0, 40.0, 80.0, 120.0, 160.0, 200.0],
            [0.0, 40.0, 80.0, 120.0, 160.0, 200.0],
            [0.0, 40.0, 80.0, 120.0, 160.0, 200.0],
            [0.0, 40.0, 80.0, 120.0, 160.0, 200.0],
            [0.0, 40.0, 80.0, 120.0, 160.0, 200.0],
            [0.0, 40.0, 80.0, 120.0, 160.0, 200.0],
            [0.0, 40.0, 80.0, 120.0, 160.0, 200.0],
        ]
    )
    assert np.all(np.equal(output_x, expected))

    # test batch size - vectorizing scenarios at first dim with same filter
    multi_input_x = np.tile(np.expand_dims(input_x, 0), (3, 1, 1))
    output_x = adstock_process(multi_input_x, adstock_filters)
    for idx in range(3):
        assert np.all(np.equal(output_x[idx], expected))

    # redo this with identical filter
    # 1 step and 100% weight should mean input is identical to output
    input_x = np.full((10, 6), np.arange(0, 6)).astype(float)
    adstock_filters = np.ones((6, 1))
    output_x = adstock_process(input_x, adstock_filters)
    assert np.all(np.equal(output_x, input_x))

    # test batch size - vectorizing scenarios at first dim with same filter
    multi_input_x = np.tile(np.expand_dims(input_x, 0), (3, 1, 1))
    output_x = adstock_process(multi_input_x, adstock_filters)
    for idx in range(3):
        assert np.all(np.equal(output_x[idx], input_x))
