from typing import Callable
from unittest import TestCase

import torch
from gradescope_utils.autograder_utils.decorators import partial_credit, visibility

from homeworks.intro_pytorch.losses import CrossEntropyLossLayer


class TestCrossEntropy(TestCase):
    @visibility("visible")
    @partial_credit(3)
    def test_crossentropy(self, set_score: Callable[[int], None]):
        try:
            layer = CrossEntropyLossLayer()
            x = torch.tensor(
                [
                    [0.3432, 0.4780, 0.1788],
                    [0.4121, 0.0274, 0.5604],
                    [0.2109, 0.5084, 0.2807],
                    [0.7671, 0.9303, -0.6973],
                    [0.1812, 0.2905, 0.5283],
                ]
            )
            target = torch.tensor([0, 1, 2, 1, 2]).long()
            expected = 1.32949

            actual = layer(x, target)
            torch.testing.assert_allclose(actual, expected, rtol=1e-2, atol=1e-4)
            set_score(1)
        except:  # noqa: E722
            raise
