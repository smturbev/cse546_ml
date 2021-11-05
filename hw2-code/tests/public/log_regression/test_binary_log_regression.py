from typing import Callable
from unittest import TestCase, TestLoader, TestSuite
from unittest.mock import Mock

import numpy as np
from gradescope_utils.autograder_utils.decorators import partial_credit, visibility

from homeworks.log_regression.binary_log_regression import BinaryLogReg


class TestBinaryLogReg(TestCase):
    @property
    def example_weight(self):
        return np.linspace(0, 1, 784) * np.tile([-1, 1], reps=784 // 2)

    @visibility("visible")
    @partial_credit(1)
    def test_mu_ones(self, set_score: Callable[[int], None]):
        try:
            weight, bias, X, y, expected = (
                np.ones(784),
                1.0,
                np.ones((30, 784)),
                np.ones(30),
                np.ones(30),
            )
            model = BinaryLogReg()
            model.weight = weight
            model.bias = bias

            actual = model.mu(X, y)

            np.testing.assert_array_almost_equal(actual, expected)
            set_score(1)
        except:  # noqa: E722
            raise

    @visibility("visible")
    @partial_credit(1)
    def test_mu_medium(self, set_score: Callable[[int], None]):
        try:
            weight = self.example_weight
            bias = 1.0
            X = np.linspace(0, 1, num=30 * 784).reshape((30, 784))
            y = np.ones(30)
            expected = np.array(
                [
                    0.7343,
                    0.7376,
                    0.7408,
                    0.744,
                    0.7471,
                    0.7503,
                    0.7534,
                    0.7565,
                    0.7595,
                    0.7626,
                    0.7656,
                    0.7686,
                    0.7715,
                    0.7745,
                    0.7774,
                    0.7802,
                    0.7831,
                    0.7859,
                    0.7887,
                    0.7915,
                    0.7942,
                    0.7969,
                    0.7996,
                    0.8023,
                    0.8049,
                    0.8075,
                    0.8101,
                    0.8126,
                    0.8152,
                    0.8177,
                ],
            )

            model = BinaryLogReg()
            model.weight = weight
            model.bias = bias

            actual = model.mu(X, y)

            np.testing.assert_array_almost_equal(actual, expected, decimal=4)
            set_score(1)
        except:  # noqa: E722
            raise

    @visibility("visible")
    @partial_credit(1)
    def test_loss(self, set_score: Callable[[int], None]):
        try:
            weight = self.example_weight
            bias = 1.0
            X = np.linspace(0, 1, num=30 * 784).reshape((30, 784))
            y = np.tile([-1, 1], reps=15)
            expected = 1.13846

            model = BinaryLogReg()
            model.weight = weight
            model.bias = bias

            actual = model.loss(X, y)

            np.testing.assert_almost_equal(actual, expected, decimal=4)
            set_score(1)
        except:  # noqa: E722
            raise

    @visibility("visible")
    @partial_credit(1)
    def test_gradient_J_weight(self, set_score: Callable[[int], None]):
        try:
            weight = np.linspace(0, 1, num=784)
            bias = 1.0
            X = np.linspace(0, 1, num=30 * 784).reshape((30, 784))
            y = np.tile([-1, 1], reps=15)
            expected = np.linspace(0.23334325, 0.2519893, 784,)

            model = BinaryLogReg()
            model.weight = weight
            model.bias = bias

            actual = model.gradient_J_weight(X, y)

            np.testing.assert_array_almost_equal(actual, expected, decimal=3)
            set_score(1)
        except:  # noqa: E722
            raise

    @visibility("visible")
    @partial_credit(1)
    def test_gradient_J_bias(self, set_score: Callable[[int], None]):
        try:
            weight = self.example_weight
            bias = 1.0
            X = np.linspace(0, 1, num=30 * 784).reshape((30, 784))
            y = np.tile([-1, 1], reps=15)
            expected = -0.2777953

            model = BinaryLogReg()
            model.weight = weight
            model.bias = bias

            actual = model.gradient_J_bias(X, y)

            np.testing.assert_almost_equal(actual, expected, decimal=4)
            set_score(1)
        except:  # noqa: E722
            raise

    @visibility("visible")
    @partial_credit(1)
    def test_predict(self, set_score: Callable[[int], None]):
        try:
            weight = self.example_weight
            bias = 0.0
            X = np.linspace(0, 1, num=30 * 784).reshape((30, 784)) - 0.5
            expected = np.array(
                [
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ]
            )

            model = BinaryLogReg()
            model.weight = weight
            model.bias = bias

            actual = model.predict(X)

            np.testing.assert_array_almost_equal(actual, expected, decimal=3)
            set_score(1)
        except:  # noqa: E722
            raise

    @visibility("visible")
    @partial_credit(1)
    def test_misclassification_error(self, set_score: Callable[[int], None]):
        try:
            weight = self.example_weight
            bias = 0.0
            X = np.linspace(0, 1, num=30 * 784).reshape((30, 784)) - 0.5
            y = np.tile([-1, 1], reps=15)
            expected = 0.467

            model = BinaryLogReg()
            model.weight = weight
            model.bias = bias

            actual = model.misclassification_error(X, y)

            np.testing.assert_array_almost_equal(actual, expected, decimal=3)
            set_score(1)
        except:  # noqa: E722
            raise

    @visibility("visible")
    @partial_credit(1)
    def test_step(self, set_score: Callable[[int], None]):
        try:
            weight = self.example_weight
            bias = 0.0
            X = np.linspace(0, 1, num=30 * 784).reshape((30, 784)) - 0.5
            y = np.tile([-1, 1], reps=15)
            expected_weight = np.copy(weight) - np.ones(784)
            expected_bias = -1.0
            learning_rate = 1.0

            model = BinaryLogReg()
            model.gradient_J_bias = Mock(return_value=1.0)
            model.gradient_J_weight = Mock(return_value=np.ones(784))
            model.weight = weight
            model.bias = bias

            model.step(X, y, learning_rate=learning_rate)

            np.testing.assert_almost_equal(model.bias, expected_bias, decimal=4)
            np.testing.assert_array_almost_equal(
                model.weight, expected_weight, decimal=4
            )
            set_score(1)
        except:  # noqa: E722
            raise


# Create a Suite for this problem
suite_binary_log_reg = TestLoader().loadTestsFromTestCase(TestBinaryLogReg)

BinaryLogRegressionTestSuite = TestSuite([suite_binary_log_reg])
