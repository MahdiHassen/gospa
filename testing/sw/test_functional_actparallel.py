"""Functional tests for the activation-parallel, stationary-weight PE."""

import contextlib
import io
import os
import sys
import unittest


SW_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "sw")
)
sys.path.insert(0, SW_DIR)

import functional as fm  # noqa: E402


class TestActivationParallelPE(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fm._VERBOSE = False

    def test_stationary_weight_full_and_tail_batches(self):
        # PID 1 has five entries: one full 4-wide batch plus a one-entry tail.
        # PID 3 starts a new stationary-weight group.
        fifo_b = [
            (2, 0, 1),
            (4, 1, 1),
            (-1, 0, 1),
            (3, 2, 1),
            (5, 3, 1),
            (7, 0, 3),
            (-3, 2, 3),
        ]
        sparse_weights = [(1, 3), (3, -2)]
        expected = {0: -11, 1: 12, 2: 15, 3: 15}

        self.assertEqual(
            fm.pe_process_actparallel(
                fifo_b, sparse_weights, feed_width=4
            ),
            expected,
        )

    def test_feed_width_changes_schedule_not_values(self):
        fifo_b = [
            (2, 0, 1),
            (4, 1, 1),
            (-1, 0, 1),
            (3, 2, 1),
            (5, 3, 1),
            (7, 0, 3),
            (-3, 2, 3),
        ]
        sparse_weights = [(1, 3), (3, -2)]
        expected = {0: -11, 1: 12, 2: 15, 3: 15}

        for feed_width in (1, 2, 3, 4, 8):
            with self.subTest(feed_width=feed_width):
                self.assertEqual(
                    fm.pe_process_actparallel(
                        fifo_b, sparse_weights, feed_width=feed_width
                    ),
                    expected,
                )

    def test_v1_pipeline_matches_dense_convolution(self):
        activation = [
            [1, 0, 2, 0],
            [0, 3, 0, 4],
            [5, 0, 6, 0],
            [0, 7, 0, 8],
        ]
        kernel = [[2, -1], [3, 4]]
        expected = fm.conv2d_reference(activation, kernel, stride=1)

        with contextlib.redirect_stdout(io.StringIO()):
            outputs = fm.goSPA_run(
                activation,
                [kernel],
                stride=1,
                num_pes=1,
                num_mults=4,
                interpretation="v1",
            )

        self.assertEqual(outputs[0], expected)

    def test_empty_stream(self):
        self.assertEqual(
            fm.pe_process_actparallel([], [(1, 3)], feed_width=4), {}
        )

    def test_invalid_inputs_raise(self):
        with self.assertRaisesRegex(ValueError, "feed_width"):
            fm.pe_process_actparallel([], [], feed_width=0)

        with self.assertRaisesRegex(ValueError, "no stationary weight"):
            fm.pe_process_actparallel([(2, 0, 1)], [(0, 3)], feed_width=4)

        with self.assertRaisesRegex(ValueError, "PID-ordered"):
            fm.pe_process_actparallel(
                [(2, 0, 3), (4, 1, 1)], [(1, 3), (3, -2)], feed_width=4
            )

        with self.assertRaisesRegex(ValueError, "duplicate sparse weight"):
            fm.pe_process_actparallel([], [(1, 3), (1, 4)], feed_width=4)


if __name__ == "__main__":
    unittest.main()
